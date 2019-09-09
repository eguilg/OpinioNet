from pytorch_pretrained_bert import BertTokenizer
from dataset import ReviewDataset, get_data_loaders
from model import OpinioNet

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
import os.path as osp
import pandas as pd
from dataset import ID2C, ID2P


def eval_epoch(model, dataloader):
	model.eval()
	step = 0

	result = pd.DataFrame(columns=['id', 'A', 'O', 'C', 'P'])

	pbar = tqdm(dataloader)
	cur_idx = 1
	for raw, x, _ in pbar:
		if step == len(dataloader):
			pbar.close()
			break
		rv_raw, _ = raw
		x = [item.cuda() for item in x]
		with torch.no_grad():
			probs, logits = model.forward(x)
			pred_result = model.gen_candidates(probs)
			pred_result = model.nms_filter(pred_result, 0.1)
		for b in range(len(pred_result)):
			opinions = pred_result[b]
			if len(opinions) == 0:
				result = result.append({'id': cur_idx, 'A': '_', 'O': '_', 'C': '_', 'P': '_'}, ignore_index=True)
			for opn in opinions:
				opn = opn[0]
				a_s, a_e, o_s, o_e = opn[0:4]
				c, p = opn[4:6]
				if a_s == 0:
					A = '_'
				else:
					A = rv_raw[b][a_s - 1: a_e]
				if o_s == 0:
					O = '_'
				else:
					O = rv_raw[b][o_s - 1: o_e]
				C = ID2C[c]
				P = ID2P[p]
				result = result.append({'id': cur_idx, 'A': A, 'O': O, 'C': C, 'P': P}, ignore_index=True)
			cur_idx += 1

		step += 1
	return result


if __name__ == '__main__':
	EP = 100
	SAVING_DIR = '../models/'
	tokenizer = BertTokenizer.from_pretrained('/home/zydq/.torch/models/bert/chinese-bert_chinese_wwm_pytorch',
											  do_lower_case=True)
	test_dataset = ReviewDataset('../data/TEST/Test_reviews.csv', None, tokenizer)
	test_loader = DataLoader(test_dataset, 12, collate_fn=test_dataset.batchify, shuffle=False, num_workers=5)

	model = OpinioNet.from_pretrained('/home/zydq/.torch/models/bert/chinese-bert_chinese_wwm_pytorch')
	model.load_state_dict(torch.load('../models/best_bert_model'))
	model.cuda()
	result = eval_epoch(model, test_loader)
	import time
	result.to_csv('../submit/result-'+str(round(time.time())) + '.csv', header=False, index=False)
	print(len(result['id'].unique()))