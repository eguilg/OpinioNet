from pytorch_pretrained_bert import BertTokenizer
from dataset import ReviewDataset, get_data_loaders_cv, ID2LAPTOP, ID2P
from lr_scheduler import GradualWarmupScheduler, ReduceLROnPlateau
from model import OpinioNet

import torch
from torch.optim import Adam

from tqdm import tqdm
import os.path as osp
import numpy as np
import pandas as pd
import copy

def f1_score(P, G, S):
	pr = S / P
	rc = S / G
	f1 = 2 * pr * rc / (pr + rc)
	return f1, pr, rc


def evaluate_sample(gt, pred):
	gt = set(gt)
	pred = set(pred)
	p = len(pred)
	g = len(gt)
	s = len(gt.intersection(pred))
	return p, g, s


def gen_submit(ret, raw):
	result = pd.DataFrame(
		columns=['id', 'AspectTerms', 'A_start', 'A_end', 'OpinionTerms', 'O_start', 'O_end', 'Categories',
				 'Polarities'])
	cur_idx = 1
	for i, opinions in enumerate(ret):

		if len(opinions) == 0:
			result.loc[result.shape[0]] = {'id': cur_idx,
										   'AspectTerms': '_', 'A_start': ' ', 'A_end': ' ',
										   'OpinionTerms': '_', 'O_start': ' ', 'O_end': ' ',
										   'Categories': '_', 'Polarities': '_'}

		for j, (opn, score) in enumerate(opinions):
			a_s, a_e, o_s, o_e = opn[0:4]
			c, p = opn[4:6]
			if a_s == 0:
				A = '_'
				a_s = ' '
				a_e = ' '
			else:
				A = raw[i][a_s - 1: a_e]
				a_s = str(a_s - 1)
				a_e = str(a_e)
			if o_s == 0:
				O = '_'
				o_s = ' '
				o_e = ' '
			else:
				O = raw[i][o_s - 1: o_e]
				o_s = str(o_s - 1)
				o_e = str(o_e)
			C = ID2LAPTOP[c]
			P = ID2P[p]
			result.loc[result.shape[0]] = {'id': cur_idx,
										   'AspectTerms': A, 'A_start': a_s, 'A_end': a_e,
										   'OpinionTerms': O, 'O_start': o_s, 'O_end': o_e,
										   'Categories': C, 'Polarities': P}
		cur_idx += 1
	return result


import json
import argparse
from config import PRETRAINED_MODELS
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--base_model', type=str, default='roberta')
	parser.add_argument('--bs', type=int, default=12)
	args = parser.parse_args()
	FOLDS = 5
	THRESH_DIR = '../models/thresh_dict.json'
	model_config = PRETRAINED_MODELS[args.base_model]
	print(model_config)

	if osp.isfile(THRESH_DIR):
		with open(THRESH_DIR, 'r', encoding='utf-8') as f:
			thresh_dict = json.load(f)
	else:
		thresh_dict = {}

	tokenizer = BertTokenizer.from_pretrained(model_config['path'], do_lower_case=True)
	cv_loaders, val_idxs = get_data_loaders_cv(rv_path='../data/TRAIN/Train_laptop_reviews.csv',
									 lb_path='../data/TRAIN/Train_laptop_labels.csv',
									 tokenizer=tokenizer,
									 batch_size=args.bs,
									 type='laptop',
									 folds=FOLDS,
									 return_val_idxs=True)

	PRED = []
	LB, GT = [], []
	VAL_IDX = []
	for cv_idx, (train_loader, val_loader) in enumerate(cv_loaders):
		model = OpinioNet.from_pretrained(model_config['path'], version=model_config['version'], focal=model_config['focal'])
		model.load_state_dict(torch.load('../models/' + model_config['name']+'_cv'+str(cv_idx)))
		model.cuda()
		model.eval()
		thresh = thresh_dict[model_config['name']+'_cv'+str(cv_idx)]['thresh']
		VAL_IDX.extend(val_idxs[cv_idx])

		for idx, ((rv_raw, lb_raw), x, y) in enumerate(val_loader):
			x = [item.cuda() for item in x]
			y = [item.cuda() for item in y]
			with torch.no_grad():
				probs, logits = model.forward(x, 'laptop')
				loss = model.loss(logits, y)
				pred_result = model.gen_candidates(probs)
				pred_result = model.nms_filter(pred_result, thresh)
			PRED.extend(pred_result)
			LB.extend(lb_raw)
			GT.extend(rv_raw)

		del model

	P, G, S = 0, 0, 0
	for b in range(len(PRED)):
		gt = LB[b]
		pred = [x[0] for x in PRED[b]]
		p, g, s = evaluate_sample(gt, pred)

		P += p
		G += g
		S += s
	f1, pr, rc = f1_score(P, G, S)
	print("f1 %.5f, pr %.5f, rc %.5f" % (f1, pr, rc))

	ZZ = list(zip(VAL_IDX, PRED, GT))

	ZZ.sort(key=lambda x: x[0])

	PRED = [p[1] for p in ZZ]
	GT = [p[2] for p in ZZ]
	result = gen_submit(PRED, GT)

	result.to_csv('../testResults/' + model_config['name'] + '.csv', header=True, index=False)
	print(len(result['id'].unique()), result.shape[0])








