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

from collections import Counter
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


def eval_epoch(model, dataloader, th):
	model.eval()
	step = 0
	result = []
	pbar = tqdm(dataloader)
	for raw, x, _ in pbar:
		if step == len(dataloader):
			pbar.close()
			break
		rv_raw, _ = raw
		x = [item.cuda() for item in x]
		with torch.no_grad():
			probs, logits = model.forward(x, 'laptop')
			pred_result = model.gen_candidates(probs)
			pred_result = model.nms_filter(pred_result, th)

		result += pred_result

		step += 1
	return result


def accum_result(old, new):
	if old is None:
		return new
	for i in range(len(old)):
		merged = Counter(dict(old[i])) + Counter(dict(new[i]))
		old[i] = list(merged.items())
	return old


def average_result(result, num):
	for i in range(len(result)):
		for j in range(len(result[i])):
			result[i][j] = (result[i][j][0], result[i][j][1] / num)
	return result

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
	parser.add_argument('--bs', type=int, default=12)
	args = parser.parse_args()
	FOLDS = 5
	THRESH_DIR = '../models/thresh_dict.json'


	with open(THRESH_DIR, 'r', encoding='utf-8') as f:
		thresh_dict = json.load(f)

	tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODELS['roberta']['path'], do_lower_case=True)
	cv_loader, val_idxs = get_data_loaders_cv(rv_path='../data/TRAIN/Train_laptop_reviews.csv',
									 lb_path='../data/TRAIN/Train_laptop_labels.csv',
									 tokenizer=tokenizer,
									 batch_size=args.bs,
									 type='laptop',
									 folds=FOLDS,
									 return_val_idxs=True)
	VAL_IDX = []
	LB, GT = [], []
	for idxs in val_idxs:
		VAL_IDX.extend(idxs)
	for train, val in cv_loader:
		for ((rv_raw, lb_raw), x, y)  in val:
			LB.extend(lb_raw)
			GT.extend(rv_raw)
	tokenizers = dict([(model_name,
					BertTokenizer.from_pretrained(model_config['path'], do_lower_case=True)
				) for model_name, model_config in PRETRAINED_MODELS.items()])
	# print(tokenizers)

	cv_loaders = dict([(model_name,
							get_data_loaders_cv(rv_path='../data/TRAIN/Train_laptop_reviews.csv',
												lb_path='../data/TRAIN/Train_laptop_labels.csv',
												tokenizer=tokenizers[model_name],
												batch_size=args.bs,
												type='laptop',
												folds=FOLDS)
				) for model_name, model_config in PRETRAINED_MODELS.items()])

	PRED = []
	for cv_idx in range(FOLDS):
		cv_model_num = 0
		cvret = None
		for model_name, model_config in PRETRAINED_MODELS.items():
			tokenizer = tokenizers[model_name]
			_, val_loader = cv_loaders[model_name][cv_idx]

			try:
				model = OpinioNet.from_pretrained(model_config['path'], version=model_config['version'],
												  focal=model_config['focal'])
				weight_name = model_config['name'] + '_cv' + str(cv_idx)
				weight = torch.load('../models/' + weight_name)
			except FileNotFoundError:
				continue
			print(weight_name)
			model.load_state_dict(weight)
			model.cuda()
			try:
				thresh = thresh_dict[weight_name]['thresh']
			except:
				thresh = 0.5
			cvret = accum_result(cvret, eval_epoch(model, val_loader, thresh))
			cv_model_num += 1
			del model
		cvret = average_result(cvret, cv_model_num)
		PRED.extend(cvret)

	PRED_COPY = copy.deepcopy(PRED)

	# P, G, S = 0, 0, 0
	# BEST_PRED = OpinioNet.nms_filter(PRED_COPY, 0.3)
	# for b in range(len(PRED_COPY)):
	# 	gt = LB[b]
	# 	pred = [x[0] for x in BEST_PRED[b]]
	# 	p, g, s = evaluate_sample(gt, pred)
	#
	# 	P += p
	# 	G += g
	# 	S += s
	# f1, pr, rc = f1_score(P, G, S)
	# print("f1 %.5f, pr %.5f, rc %.5f, th %.5f" % (f1, pr, rc, 0.3))

	threshs = list(np.arange(0.1, 0.9, 0.025))
	best_f1, best_pr, best_rc = 0, 0, 0
	best_thresh = 0.1
	P, G, S = 0, 0, 0
	BEST_PRED = PRED_COPY
	for th in threshs:
		P, G, S = 0, 0, 0
		PRED_COPY = copy.deepcopy(PRED)
		PRED_COPY = OpinioNet.nms_filter(PRED_COPY, th)
		for b in range(len(PRED_COPY)):
			gt = LB[b]
			pred = [x[0] for x in PRED_COPY[b]]
			p, g, s = evaluate_sample(gt, pred)

			P += p
			G += g
			S += s
		f1, pr, rc = f1_score(P, G, S)
		if f1 > best_f1:
			best_f1, best_pr, best_rc = f1, pr, rc
			best_thresh = th
			BEST_PRED = copy.deepcopy(PRED_COPY)

	print("f1 %.5f, pr %.5f, rc %.5f, th %.5f" % (best_f1, best_pr, best_rc, best_thresh))

	ZZ = list(zip(VAL_IDX, BEST_PRED, GT))
	ZZ.sort(key=lambda x: x[0])

	BEST_PRED = [p[1] for p in ZZ]
	GT = [p[2] for p in ZZ]
	result = gen_submit(BEST_PRED, GT)
	if not osp.exists('../testResults/'):
		import os
		os.mkdir('../testResults/')
	result.to_csv('../testResults/' + 'ensemble_result_label_'+ ('%.5f' % best_f1) +'.csv', header=True, index=False)
	print(len(result['id'].unique()), result.shape[0])








