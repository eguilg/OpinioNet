import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import pandas as pd
from sklearn.model_selection import KFold
import jieba
import numpy as np
from data_augmentation import data_augment
# { 硬件&性能、软件&性能、外观、使用场景、物流、服务、包装、价格、真伪、整体、其他 }
ID2C = ['包装', '成分', '尺寸', '服务', '功效', '价格', '气味', '使用体验', '物流', '新鲜度', '真伪', '整体', '其他']
ID2COMMON = ['物流', '服务', '包装', '价格', '真伪', '整体', '其他']
ID2LAPTOP = ID2COMMON + ['硬件&性能', '软件&性能', '外观', '使用场景']
ID2MAKUP = ID2COMMON + ['成分', '尺寸', '功效', '气味', '使用体验', '新鲜度']

ID2P = ['正面', '中性', '负面']


# C2ID = dict(zip(ID2C, range(len(ID2C))))
LAPTOP2ID = dict(zip(ID2LAPTOP, range(len(ID2LAPTOP))))
MAKUP2ID = dict(zip(ID2MAKUP, range(len(ID2MAKUP))))
P2ID = dict(zip(ID2P, range(len(ID2P))))

class CorpusDataset(Dataset):
	def __init__(self, corpus_path, tokenizer):
		super(CorpusDataset, self).__init__()
		corpus_df = pd.read_csv(corpus_path, encoding='utf-8')
		self.tokenizer = tokenizer
		self.samples = self._preprocess_data(corpus_df, tokenizer)

	def _preprocess_data(self, corpus_df, tokenizer):
		samples = []
		for id, rv in zip(corpus_df['id'], corpus_df['Reviews']):
			if len(rv) >= 120:
				continue
			RV = ['[CLS]']
			RV_INTERVALS = []
			rv_cut = jieba.cut(rv)
			for word in rv_cut:
				s = len(RV)
				for c in word:
					if c == ' ':
						RV.append('[unused1]')
					elif c in tokenizer.vocab:
						RV.append(c)
					else:
						RV.append('[UNK]')
				e = len(RV)
				RV_INTERVALS.append((s, e))

			RV.append('[SEP]')
			RV = tokenizer.convert_tokens_to_ids(RV)

			samples.append((rv, RV, RV_INTERVALS))
		return samples

	def batchify(self, batch_samples):
		rv_raw = []
		INPUT_IDS = []
		ATTN_MASK = []
		LM_LABEL = []
		for raw, rv, rv_intervals in batch_samples:
			rv_raw.append(raw)
			masked_rv = [_ for _ in rv]
			lm_label = [-1] * len(masked_rv)
			mask_word_num = int(len(rv_intervals) * 0.15)
			masked_word_idxs = list(np.random.choice(list(range(len(rv_intervals))), mask_word_num, False))
			for i in masked_word_idxs:
				s, e = rv_intervals[i]
				for j in range(s, e):
					lm_label[j] = masked_rv[j]
					rand = np.random.rand()
					if rand < 0.1: # 随机替换
						replace_id = np.random.choice(range(len(self.tokenizer.vocab)))
					elif rand < 0.2: # 保留原词
						replace_id = lm_label[j]
					else: #换为mask
						replace_id = self.tokenizer.vocab['[MASK]']
					masked_rv[j] = replace_id

			ATTN_MASK.append([1] * len(masked_rv))
			INPUT_IDS.append(masked_rv)
			LM_LABEL.append(lm_label)
		INPUT_IDS = torch.LongTensor(pad_batch_seqs(INPUT_IDS, self.tokenizer.vocab['[PAD]']))
		ATTN_MASK = torch.LongTensor(pad_batch_seqs(ATTN_MASK, 0))
		LM_LABEL = torch.LongTensor(pad_batch_seqs(LM_LABEL, -1))

		return INPUT_IDS, ATTN_MASK, LM_LABEL


	def __getitem__(self, index):
		return self.samples[index]

	def __len__(self):
		return len(self.samples)

class ReviewDataset(Dataset):
	def __init__(self, reviews, labels, tokenizer, type='makeup'):
		super(ReviewDataset, self).__init__()
		if isinstance(reviews, str):
			reviews_df = pd.read_csv(reviews, encoding='utf-8')
		elif isinstance(reviews, pd.DataFrame):
			reviews_df = reviews
		else:
			raise TypeError("接受路径或df")
		if type == 'makeup':
			self.C2ID = MAKUP2ID
		else:
			self.C2ID = LAPTOP2ID
		labels_df = None
		if labels is not None:
			if isinstance(labels, str):
				labels_df = pd.read_csv(labels, encoding='utf-8')
			elif isinstance(labels, pd.DataFrame):
				labels_df = labels
			else:
				raise TypeError("接受路径或df")

		self.samples = self._preprocess_data(reviews_df, labels_df, tokenizer)
		self.PAD_ID = tokenizer.vocab['[PAD]']


	def __getitem__(self, index):
		return self.samples[index]

	def __len__(self):
		return len(self.samples)

	def _preprocess_data(self, reviews_df, labels_df, tokenizer):
		samples = []
		for id, rv in zip(reviews_df['id'], reviews_df['Reviews']):
			rv = rv[:120]
			RV = []
			for c in rv:
				if c == ' ':
					RV.append('[unused1]')
				elif c in tokenizer.vocab:
					RV.append(c)
				else:
					RV.append('[UNK]')

			# RV = [c if c in tokenizer.vocab else '[UNK]' for c in rv]
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

					if c in self.C2ID:
						c = self.C2ID[c]
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

				# print(LB_NUM)
				# obj_weights = 1 / sum(LB_OBJ)
				# LB_OBJ = list(map(lambda x: obj_weights if x == 1 else 0, LB_OBJ))
				LABELS = (LB_AS, LB_AE, LB_OS, LB_OE, LB_OBJ, LB_C, LB_P)
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
			# TGT_AS, TGT_AE, TGT_OS, TGT_OE, TGT_OBJ, TGT_C, TGT_P = [], [], [], [], [], [], []
			for _, RV, LABELS in batch_samples:
				for i in range(len(LABELS)):
					TARGETS[i].append(LABELS[i])

			for i in range(len(TARGETS)):
				if i == 4:
					TARGETS[i] = torch.FloatTensor(pad_batch_seqs(TARGETS[i], 0))  # OBJ for kldiv
				# elif i == len(TARGETS) - 1:
				# 	TARGETS[i] = torch.LongTensor(TARGETS[i])
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


def get_full_data_loaders(rv_path, lb_path, tokenizer, batch_size, type='makeup', shuffle=False):
	full_dataset = ReviewDataset(rv_path, lb_path, tokenizer, type)
	loader = DataLoader(full_dataset, batch_size, collate_fn=full_dataset.batchify, shuffle=shuffle, num_workers=5,
							  drop_last=False)

	return loader


def get_data_loaders_cv(rv_path, lb_path, tokenizer, batch_size, type='makeup', folds=5, return_val_idxs=False):
	full_dataset = ReviewDataset(rv_path, lb_path, tokenizer, type)

	kf = KFold(n_splits=folds, shuffle=True, random_state=502)
	folds = kf.split(full_dataset)
	cv_loaders = []
	val_idxs = []
	for train_idx, val_idx in folds:
		train_loader = DataLoader([full_dataset.samples[i] for i in train_idx], batch_size,
								  collate_fn=full_dataset.batchify, shuffle=True, num_workers=5, drop_last=False)
		val_loader = DataLoader([full_dataset.samples[i] for i in val_idx], batch_size,
								collate_fn=full_dataset.batchify, shuffle=False, num_workers=5, drop_last=False)
		cv_loaders.append((train_loader, val_loader))
		val_idxs.append(val_idx)

	if return_val_idxs:
		return cv_loaders, val_idxs

	return cv_loaders


def get_aug_data_loaders_cv(rv_path, lb_path, tokenizer, batch_size, type='makeup', folds=5):
	# full_dataset = ReviewDataset(rv_path, lb_path, tokenizer, type)
	rv_df = pd.read_csv(rv_path, encoding='utf-8')
	lb_df = pd.read_csv(lb_path, encoding='utf-8')
	kf = KFold(n_splits=folds, shuffle=True, random_state=502)
	folds = kf.split(range(rv_df.shape[0]))
	for train_idx, val_idx in folds:
		train_rv_df = rv_df.iloc[train_idx].copy()
		train_lb_df = lb_df[lb_df['id'].isin(train_rv_df['id'])].copy()
		val_rv_df = rv_df.iloc[val_idx].copy()
		val_lb_df = lb_df[lb_df['id'].isin(val_rv_df['id'])].copy()

		train_rv_aug_df, train_lb_aug_df = data_augment(train_rv_df, train_lb_df, 1)

		print(train_rv_aug_df.shape[0])
		print(train_rv_df.shape[0])
		train_rv_df['id'] += train_rv_aug_df.shape[0]
		train_lb_df['id'] += train_rv_aug_df.shape[0]
		train_rv_aug_df = train_rv_aug_df.append(train_rv_df, ignore_index=True)
		train_lb_aug_df = train_lb_aug_df.append(train_lb_df, ignore_index=True)

		train_dataset = ReviewDataset(train_rv_aug_df, train_lb_aug_df, tokenizer, type)
		val_dataset = ReviewDataset(val_rv_df, val_lb_df, tokenizer, type)


		train_loader = DataLoader(train_dataset, batch_size,
								  collate_fn=train_dataset.batchify, shuffle=True, num_workers=5, drop_last=False)
		val_loader = DataLoader(val_dataset, batch_size,
								collate_fn=val_dataset.batchify, shuffle=False, num_workers=5, drop_last=False)
		yield train_loader, val_loader


def get_data_loaders_round2(tokenizer, batch_size, val_split=0.15):
	makeup_rv1 = ReviewDataset('../data/TRAIN/Train_reviews.csv', '../data/TRAIN/Train_labels.csv', tokenizer)
	makeup_rv2 = ReviewDataset('../data/TRAIN/Train_makeup_reviews.csv', '../data/TRAIN/Train_makeup_labels.csv', tokenizer)
	makeup_rv = ConcatDataset([makeup_rv1, makeup_rv2])
	laptop_rv = ReviewDataset('../data/TRAIN/Train_laptop_reviews.csv', '../data/TRAIN/Train_laptop_labels.csv', tokenizer, type='laptop')


	laptop_corpus1 = CorpusDataset('../data/TEST/Test_reviews.csv', tokenizer)
	laptop_corpus2 = CorpusDataset('../data/TRAIN/Train_laptop_corpus.csv', tokenizer)
	laptop_corpus3 = CorpusDataset('../data/TRAIN/Train_laptop_reviews.csv', tokenizer)
	makeup_corpus1 = CorpusDataset('../data/TEST/Test_reviews1.csv', tokenizer)
	makeup_corpus2 = CorpusDataset('../data/TRAIN/Train_reviews.csv', tokenizer)
	makeup_corpus3 = CorpusDataset('../data/TRAIN/Train_makeup_reviews.csv', tokenizer)

	corpus_rv = ConcatDataset([laptop_corpus1, laptop_corpus2, laptop_corpus3, makeup_corpus1, makeup_corpus2, makeup_corpus3])
	corpus_loader = DataLoader(corpus_rv, batch_size, collate_fn=laptop_corpus1.batchify, shuffle=True, num_workers=5,
							   drop_last=False)

	makeup_train_size = int(len(makeup_rv) * (1 - val_split))
	makeup_val_size = len(makeup_rv) - makeup_train_size
	torch.manual_seed(502)
	makeup_train, makeup_val = random_split(makeup_rv, [makeup_train_size, makeup_val_size])
	makeup_train_loader = DataLoader(makeup_train, batch_size // 2, collate_fn=makeup_rv1.batchify, shuffle=True, num_workers=5,
							  drop_last=False)
	makeup_val_loader = DataLoader(makeup_val, batch_size, collate_fn=makeup_rv1.batchify, shuffle=False, num_workers=5,
							drop_last=False)

	laptop_train_size = int(len(laptop_rv) * (1 - val_split))
	laptop_val_size = len(laptop_rv) - laptop_train_size
	torch.manual_seed(502)
	laptop_train, laptop_val = random_split(laptop_rv, [laptop_train_size, laptop_val_size])
	laptop_train_loader = DataLoader(laptop_train, batch_size // 2, collate_fn=laptop_rv.batchify, shuffle=True,
									 num_workers=5,
									 drop_last=False)
	laptop_val_loader = DataLoader(laptop_val, batch_size, collate_fn=laptop_rv.batchify, shuffle=False, num_workers=5,
								   drop_last=False)

	return makeup_train_loader, makeup_val_loader, laptop_train_loader, laptop_val_loader, corpus_loader


def get_pretrain_loaders(tokenizer, batch_size, val_split=0.15):
	makeup_rv1 = ReviewDataset('../data/TRAIN/Train_reviews.csv', '../data/TRAIN/Train_labels.csv', tokenizer)
	makeup_rv2 = ReviewDataset('../data/TRAIN/Train_makeup_reviews.csv', '../data/TRAIN/Train_makeup_labels.csv',
							   tokenizer)
	makeup_rv = ConcatDataset([makeup_rv1, makeup_rv2])
	laptop_corpus1 = CorpusDataset('../data/TEST/Test_reviews.csv', tokenizer)
	laptop_corpus2 = CorpusDataset('../data/TRAIN/Train_laptop_corpus.csv', tokenizer)
	laptop_corpus3 = CorpusDataset('../data/TRAIN/Train_laptop_reviews.csv', tokenizer)
	makeup_corpus1 = CorpusDataset('../data/TEST/Test_reviews1.csv', tokenizer)
	makeup_corpus2 = CorpusDataset('../data/TRAIN/Train_reviews.csv', tokenizer)
	makeup_corpus3 = CorpusDataset('../data/TRAIN/Train_makeup_reviews.csv', tokenizer)

	corpus_rv = ConcatDataset(
		[laptop_corpus1, laptop_corpus2, laptop_corpus3, makeup_corpus1, makeup_corpus2, makeup_corpus3])
	corpus_loader = DataLoader(corpus_rv, batch_size, collate_fn=laptop_corpus1.batchify, shuffle=True, num_workers=5,
							   drop_last=False)
	makeup_train_size = int(len(makeup_rv) * (1 - val_split))
	makeup_val_size = len(makeup_rv) - makeup_train_size
	torch.manual_seed(502)
	makeup_train, makeup_val = random_split(makeup_rv, [makeup_train_size, makeup_val_size])
	makeup_train_loader = DataLoader(makeup_train, batch_size, collate_fn=makeup_rv1.batchify, shuffle=True,
									 num_workers=5,
									 drop_last=False)
	makeup_val_loader = DataLoader(makeup_val, batch_size, collate_fn=makeup_rv1.batchify, shuffle=False, num_workers=5,
								   drop_last=False)

	return makeup_train_loader, makeup_val_loader, corpus_loader


def get_pretrain2_loaders(tokenizer, batch_size, val_split=0.15):
	makeup_rv1 = ReviewDataset('../data/TRAIN/Train_reviews.csv', '../data/TRAIN/Train_labels.csv', tokenizer)
	makeup_rv2 = ReviewDataset('../data/TRAIN/Train_makeup_reviews.csv', '../data/TRAIN/Train_makeup_labels.csv',
							   tokenizer)
	makeup_rv = ConcatDataset([makeup_rv1, makeup_rv2])
	makeup_loader = DataLoader(makeup_rv, batch_size, collate_fn=makeup_rv1.batchify, shuffle=True,
									 num_workers=5,
									 drop_last=False)

	laptop_rv = ReviewDataset('../data/TRAIN/Train_laptop_corpus.csv', '../data/TRAIN/Train_laptop_corpus_labels.csv', tokenizer, 'laptop')
	laptop_val_rv = ReviewDataset('../data/TRAIN/Train_laptop_reviews.csv', '../data/TRAIN/Train_laptop_labels.csv', tokenizer, 'laptop')

	laptop_loader = DataLoader(laptop_rv, batch_size, collate_fn=laptop_rv.batchify, shuffle=True,
									 num_workers=5,
									 drop_last=False)

	laptop_val_loader = DataLoader(laptop_val_rv, batch_size, collate_fn=laptop_val_rv.batchify, shuffle=False,
									 num_workers=5,
									 drop_last=False)

	laptop_corpus1 = CorpusDataset('../data/TEST/Test_reviews.csv', tokenizer)
	laptop_corpus2 = CorpusDataset('../data/TRAIN/Train_laptop_corpus.csv', tokenizer)
	laptop_corpus3 = CorpusDataset('../data/TRAIN/Train_laptop_reviews.csv', tokenizer)
	makeup_corpus1 = CorpusDataset('../data/TEST/Test_reviews1.csv', tokenizer)
	makeup_corpus2 = CorpusDataset('../data/TRAIN/Train_reviews.csv', tokenizer)
	makeup_corpus3 = CorpusDataset('../data/TRAIN/Train_makeup_reviews.csv', tokenizer)

	corpus_rv = ConcatDataset(
		[laptop_corpus1, laptop_corpus2, laptop_corpus3, makeup_corpus1, makeup_corpus2, makeup_corpus3])
	corpus_loader = DataLoader(corpus_rv, batch_size, collate_fn=laptop_corpus1.batchify, shuffle=True, num_workers=5,
							   drop_last=False)

	return makeup_loader, laptop_loader, laptop_val_loader, corpus_loader

def get_makeup_full_loaders(tokenizer, batch_size):
	makeup_rv1 = ReviewDataset('../data/TRAIN/Train_reviews.csv', '../data/TRAIN/Train_labels.csv', tokenizer)
	makeup_rv2 = ReviewDataset('../data/TRAIN/Train_makeup_reviews.csv', '../data/TRAIN/Train_makeup_labels.csv',
							   tokenizer)
	makeup_rv = ConcatDataset([makeup_rv1, makeup_rv2])
	makeup_train_loader = DataLoader(makeup_rv, batch_size, collate_fn=makeup_rv1.batchify, shuffle=True,
									 num_workers=5,
									 drop_last=False)

	return makeup_train_loader

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
