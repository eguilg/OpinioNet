from pytorch_pretrained_bert import BertTokenizer
from dataset import ReviewDataset, get_data_loaders
from model import OpinioNet

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
import os.path as osp
import pandas as pd
from dataset import ID2C, ID2P
from collections import Counter


def eval_epoch(model, dataloader):
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
			probs, logits = model.forward(x)
			pred_result = model.gen_candidates(probs)
			pred_result = model.nms_filter(pred_result, 0)

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
	result = pd.DataFrame(columns=['id', 'A', 'O', 'C', 'P'])
	cur_idx = 1
	for i, opinions in enumerate(ret):

		if len(opinions) == 0:
			result = result.append({'id': cur_idx, 'A': '_', 'O': '_', 'C': '_', 'P': '_'}, ignore_index=True)

		for j, (opn, score) in enumerate(opinions):
			a_s, a_e, o_s, o_e = opn[0:4]
			c, p = opn[4:6]
			if a_s == 0:
				A = '_'
			else:
				A = raw[i][a_s - 1: a_e]
			if o_s == 0:
				O = '_'
			else:
				O = raw[i][o_s - 1: o_e]
			C = ID2C[c]
			P = ID2P[p]
			result = result.append({'id': cur_idx, 'A': A, 'O': O, 'C': C, 'P': P}, ignore_index=True)
		cur_idx += 1
	return result


if __name__ == '__main__':
	THRESH = 0.10
	SAVING_DIR = '../models/'
	MODELS = [
		'best_bert_model_774',
		'best_bert_model_77',
		'best_bert_model_cv0',
		'best_bert_model_cv1',
		'best_bert_model_cv2',
		'best_bert_model_cv3',
		'best_bert_model_cv4'
	]
	tokenizer = BertTokenizer.from_pretrained('/home/zydq/.torch/models/bert/chinese-bert_chinese_wwm_pytorch',
											  do_lower_case=True)
	test_dataset = ReviewDataset('../data/TEST/Test_reviews.csv', None, tokenizer)
	test_loader = DataLoader(test_dataset, 12, collate_fn=test_dataset.batchify, shuffle=False, num_workers=5)

	ret = None
	for name in MODELS:
		model_path = osp.join(SAVING_DIR, name)
		model = OpinioNet.from_pretrained('/home/zydq/.torch/models/bert/chinese-bert_chinese_wwm_pytorch')
		model.load_state_dict(torch.load(model_path))
		model.cuda()
		ret = accum_result(ret, eval_epoch(model, test_loader))
		del model
	ret = average_result(ret, len(MODELS))
	ret = OpinioNet.nms_filter(ret, THRESH)
	raw = [s[0][0] for s in test_dataset.samples]
	result = gen_submit(ret, raw)
	import time

	result.to_csv('../submit/ensemble-' + str(round(time.time())) + '.csv', header=False, index=False)
	print(len(result['id'].unique()), result.shape[0])
