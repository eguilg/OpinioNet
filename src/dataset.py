import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import pandas as pd
from sklearn.model_selection import KFold

ID2C = ['包装', '成分', '尺寸', '服务', '功效', '价格', '气味', '使用体验', '物流', '新鲜度', '真伪', '整体', '其他']
ID2P = ['正面', '中性', '负面']

C2ID = dict(zip(ID2C, range(len(ID2C))))
P2ID = dict(zip(ID2P, range(len(ID2P))))


class ReviewDataset(Dataset):
	def __init__(self, reviews_path, labels_path, tokenizer):
		super(ReviewDataset, self).__init__()
		reviews_df = pd.read_csv(reviews_path, encoding='utf-8')
		labels_df = None
		if labels_path is not None:
			labels_df = pd.read_csv(labels_path, encoding='utf-8')

		self.samples = self._preprocess_data(reviews_df, labels_df, tokenizer)
		self.PAD_ID = tokenizer.vocab['[PAD]']

	def __getitem__(self, index):
		return self.samples[index]

	def __len__(self):
		return len(self.samples)

	@staticmethod
	def _preprocess_data(reviews_df, labels_df, tokenizer):
		samples = []
		for id, rv in zip(reviews_df['id'], reviews_df['Reviews']):
			RV = []
			for c in rv:
				if c == ' ':
					RV.append('[unused1]')
				elif c in tokenizer.vocab:
					RV.append(c)
				else:
					RV.append('[UNK]')

			RV = [c if c in tokenizer.vocab else '[UNK]' for c in rv]
			RV = ['[CLS]'] + RV + ['[SEP]']
			RV = tokenizer.convert_tokens_to_ids(RV)

			if labels_df is not None:
				lbs = labels_df[labels_df['id'] == id]
				# A_start & A_end
				LB_AS = [-1] * len(RV)
				LB_AE = [-1] * len(RV)

				# O_start & O_end
				LB_OS = [-1] * len(RV)
				LB_OE = [-1] * len(RV)

				# Objectiveness
				LB_OBJ = [0] * len(RV)

				# Categories
				LB_C = [-1] * len(RV)

				# Polarities
				LB_P = [-1] * len(RV)

				lb_raw = []
				for i in range(len(lbs)):
					lb = lbs.iloc[i]
					a_s = lb['A_start'].strip()
					a_e = lb['A_end'].strip()
					o_s = lb['O_start'].strip()
					o_e = lb['O_end'].strip()
					c = lb['Categories'].strip()
					p = lb['Polarities'].strip()

					if c in C2ID:
						c = C2ID[c]
					else:
						c = -1

					if p in P2ID:
						p = P2ID[p]
					else:
						p = -1
					# a和o均从1开始 0 代表CLS
					if a_s != '' and a_e != '':
						a_s, a_e = int(a_s) + 1, int(a_e)
					else:
						a_s, a_e = 0, 0
					if o_s != '' and o_e != '':
						o_s, o_e = int(o_s) + 1, int(o_e)
					else:
						o_s, o_e = 0, 0

					if a_s >= len(RV) - 1:
						a_s, a_e = 0, 0
					if o_s >= len(RV) - 1:
						o_s, o_e = 0, 0

					a_s = min(a_s, len(RV) - 2)
					a_e = min(a_e, len(RV) - 2)
					o_s = min(o_s, len(RV) - 2)
					o_e = min(o_e, len(RV) - 2)

					# print(a_s, a_e, o_s, o_e, len(RV))

					if a_s > 0:
						LB_AS[a_s: a_e + 1] = [a_s] * (a_e - a_s + 1)
						LB_AE[a_s: a_e + 1] = [a_e] * (a_e - a_s + 1)
						LB_OS[a_s: a_e + 1] = [o_s] * (a_e - a_s + 1)
						LB_OE[a_s: a_e + 1] = [o_e] * (a_e - a_s + 1)
						LB_OBJ[a_s: a_e + 1] = [1] * (a_e - a_s + 1)
						LB_C[a_s: a_e + 1] = [c] * (a_e - a_s + 1)
						LB_P[a_s: a_e + 1] = [p] * (a_e - a_s + 1)

					if o_s > 0:
						LB_AS[o_s: o_e + 1] = [a_s] * (o_e - o_s + 1)
						LB_AE[o_s: o_e + 1] = [a_e] * (o_e - o_s + 1)
						LB_OS[o_s: o_e + 1] = [o_s] * (o_e - o_s + 1)
						LB_OE[o_s: o_e + 1] = [o_e] * (o_e - o_s + 1)
						LB_OBJ[o_s: o_e + 1] = [1] * (o_e - o_s + 1)
						LB_C[o_s: o_e + 1] = [c] * (o_e - o_s + 1)
						LB_P[o_s: o_e + 1] = [p] * (o_e - o_s + 1)
					lb_raw.append((a_s, a_e, o_s, o_e, c, p))

				LB_NUM = len(lbs)
				# print(LB_NUM)
				# obj_weights = 1 / sum(LB_OBJ)
				# LB_OBJ = list(map(lambda x: obj_weights if x == 1 else 0, LB_OBJ))
				LABELS = (LB_AS, LB_AE, LB_OS, LB_OE, LB_OBJ, LB_C, LB_P, LB_NUM)
				rv = (rv, lb_raw)
			else:
				LABELS = None
				rv = (rv, None)

			samples.append((rv, RV, LABELS))
		return samples

	def batchify(self, batch_samples):
		rv_raw = []
		lb_raw = []
		IN_RV = []
		IN_ATT_MASK = []
		IN_RV_MASK = []

		for raw, RV, _ in batch_samples:
			rv_raw.append(raw[0])
			lb_raw.append(raw[1])
			IN_RV.append(RV)
			IN_ATT_MASK.append([1] * len(RV))
			IN_RV_MASK.append([0] + [1] * (len(RV) - 2) + [0])

		IN_RV = torch.LongTensor(pad_batch_seqs(IN_RV, self.PAD_ID))
		IN_ATT_MASK = torch.LongTensor(pad_batch_seqs(IN_ATT_MASK, 0))
		IN_RV_MASK = torch.LongTensor(pad_batch_seqs(IN_RV_MASK, 0))

		INPUTS = [IN_RV, IN_ATT_MASK, IN_RV_MASK]

		if batch_samples[0][2] is not None:
			TARGETS = [[] for _ in batch_samples[0][2]]
			# TGT_AS, TGT_AE, TGT_OS, TGT_OE, TGT_OBJ, TGT_C, TGT_P, TGT_NUM = [], [], [], [], [], [], [], []
			for _, RV, LABELS in batch_samples:
				for i in range(len(LABELS)):
					TARGETS[i].append(LABELS[i])

			for i in range(len(TARGETS)):
				if i == 4:
					TARGETS[i] = torch.FloatTensor(pad_batch_seqs(TARGETS[i], 0))  # OBJ for kldiv
				elif i == len(TARGETS) - 1:
					TARGETS[i] = torch.LongTensor(TARGETS[i])
				else:
					TARGETS[i] = torch.LongTensor(pad_batch_seqs(TARGETS[i], -1))  # for CE Loss ignore
		else:
			TARGETS = None

		return (rv_raw, lb_raw), INPUTS, TARGETS


def pad_batch_seqs(seqs: list, pad=None, max_len=None) -> list:
	if not max_len:
		max_len = max([len(s) for s in seqs])
	if not pad:
		pad = 0
	for i in range(len(seqs)):
		if len(seqs[i]) > max_len:
			seqs[i] = seqs[i][:max_len]
		else:
			seqs[i].extend([pad] * (max_len - len(seqs[i])))

	return seqs


def get_data_loaders(rv_path, lb_path, tokenizer, batch_size, val_split=0.15):
	full_dataset = ReviewDataset(rv_path, lb_path, tokenizer)
	train_size = int(len(full_dataset) * (1 - val_split))
	val_size = len(full_dataset) - train_size
	lengths = [train_size, val_size]
	torch.manual_seed(502)
	train_data, val_data = random_split(full_dataset, lengths)
	train_loader = DataLoader(train_data, batch_size, collate_fn=full_dataset.batchify, shuffle=True, num_workers=5,
							  drop_last=False)
	val_loader = DataLoader(val_data, batch_size, collate_fn=full_dataset.batchify, shuffle=False, num_workers=5,
							drop_last=False)

	return train_loader, val_loader


def get_data_loaders_cv(rv_path, lb_path, tokenizer, batch_size, folds=5):
	full_dataset = ReviewDataset(rv_path, lb_path, tokenizer)

	kf = KFold(n_splits=folds, shuffle=True, random_state=502)
	folds = kf.split(full_dataset)
	cv_loaders = []
	for train_idx, val_idx in folds:
		train_loader = DataLoader([full_dataset.samples[i] for i in train_idx], batch_size,
								  collate_fn=full_dataset.batchify, shuffle=True, num_workers=5, drop_last=False)
		val_loader = DataLoader([full_dataset.samples[i] for i in val_idx], batch_size,
								collate_fn=full_dataset.batchify, shuffle=True, num_workers=5, drop_last=False)
		cv_loaders.append((train_loader, val_loader))

	return cv_loaders


if __name__ == '__main__':
	from pytorch_pretrained_bert import BertTokenizer

	tokenizer = BertTokenizer.from_pretrained('/home/zydq/.torch/models/bert/chinese-bert_chinese_wwm_pytorch',
											  do_lower_case=True)
	d = ReviewDataset('../data/TRAIN/Train_reviews.csv', '../data/TRAIN/Train_labels.csv', tokenizer)
	mxl = 0
	mxn = 0
	for raw, a, b in d:
		mxl = max(len(raw), mxl)
		mxn = max(b[-1], mxn)
	print(mxl, mxn)
