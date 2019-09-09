from pytorch_pretrained_bert import BertTokenizer
from dataset import ReviewDataset, get_data_loaders
from model import OpinioNet

import torch
from torch.optim import Adam

from lr_scheduler import GradualWarmupScheduler, ReduceLROnPlateau
from tqdm import tqdm
import os.path as osp


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
	# print(p, g, s)
	return p, g, s


def train_epoch(model, dataloader, optimizer, scheduler=None):
	model.train()
	cum_loss = 0
	P, G, S = 0, 0, 0
	total_sample = 0
	step = 0
	pbar = tqdm(dataloader)
	for raw, x, y in pbar:
		if step == len(dataloader):
			pbar.close()
			break
		rv_raw, lb_raw = raw
		x = [item.cuda() for item in x]
		y = [item.cuda() for item in y]

		probs, logits = model.forward(x)
		loss = model.loss(logits, y)

		optimizer.zero_grad()
		loss.backward()
		if scheduler:
			scheduler.step()
		optimizer.step()

		pred_result = model.gen_candidates(probs)
		pred_result = model.nms_filter(pred_result, 0.1)
		cum_loss += loss.data.cpu().numpy() * len(rv_raw)
		total_sample += len(rv_raw)
		for b in range(len(pred_result)):
			gt = lb_raw[b]
			pred = [x[0] for x in pred_result[b]]
			p, g, s = evaluate_sample(gt, pred)
			P += p
			G += g
			S += s

		step += 1

	total_f1, total_pr, total_rc = f1_score(P, G, S)
	total_loss = cum_loss / total_sample

	return total_loss, total_f1, total_pr, total_rc


def eval_epoch(model, dataloader):
	model.eval()
	cum_loss = 0
	P, G, S = 0, 0, 0
	total_sample = 0
	step = 0
	pbar = tqdm(dataloader)
	for raw, x, y in pbar:
		if step == len(dataloader):
			pbar.close()
			break
		rv_raw, lb_raw = raw
		x = [item.cuda() for item in x]
		y = [item.cuda() for item in y]
		with torch.no_grad():
			probs, logits = model.forward(x)
			loss = model.loss(logits, y)
			pred_result = model.gen_candidates(probs)
			pred_result = model.nms_filter(pred_result, 0.1)

		cum_loss += loss.data.cpu().numpy() * len(rv_raw)
		total_sample += len(rv_raw)
		for b in range(len(pred_result)):
			gt = lb_raw[b]
			pred = [x[0] for x in pred_result[b]]
			p, g, s = evaluate_sample(gt, pred)

			P += p
			G += g
			S += s

		step += 1

	total_f1, total_pr, total_rc = f1_score(P, G, S)
	total_loss = cum_loss / total_sample

	return total_loss, total_f1, total_pr, total_rc


if __name__ == '__main__':
	EP = 100
	SAVING_DIR = '../models/'
	tokenizer = BertTokenizer.from_pretrained('/home/zydq/.torch/models/bert/chinese-bert_chinese_wwm_pytorch',
											  do_lower_case=True)
	train_loader, val_loader = get_data_loaders(rv_path='../data/TRAIN/Train_reviews.csv',
												lb_path='../data/TRAIN/Train_labels.csv',
												tokenizer=tokenizer,
												batch_size=12,
												val_split=0.15)

	model = OpinioNet.from_pretrained('/home/zydq/.torch/models/bert/chinese-bert_chinese_wwm_pytorch')
	model.cuda()
	optimizer = Adam(model.parameters(), lr=5e-6)
	scheduler = GradualWarmupScheduler(optimizer, total_epoch=2)
	best_val_f1 = 0
	best_val_loss = float('inf')
	for e in range(EP):

		print('Epoch [%d/%d] train:' % (e, EP))
		train_loss, train_f1, train_pr, train_rc = train_epoch(model, train_loader, optimizer, scheduler)
		print("loss %.5f, f1 %.5f, pr %.5f, rc %.5f" % (train_loss, train_f1, train_pr, train_rc))

		print('Epoch [%d/%d] eval:' % (e, EP))
		val_loss, val_f1, val_pr, val_rc = eval_epoch(model, val_loader)
		print("loss %.5f, f1 %.5f, pr %.5f, rc %.5f" % (val_loss, val_f1, val_pr, val_rc))
		if val_loss < best_val_loss:
			best_val_loss = val_loss
		if val_f1 > best_val_f1:
			best_val_f1 = val_f1
			if best_val_f1 >= 0.75:
				model_name_args = ['ep' + str(e), 'f1' + str(round(val_f1, 5))]
				model_name = '-'.join(model_name_args) + '.pth'
				saving_dir = osp.join(SAVING_DIR, 'saved_best_model')
				torch.save(model.state_dict(), saving_dir)
				print('saved best model to %s' % saving_dir)

		print('best loss %.5f' % best_val_loss)
		print('best f1 %.5f' % best_val_f1)