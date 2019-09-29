from pytorch_pretrained_bert import BertTokenizer
from dataset import ReviewDataset, get_data_loaders, get_data_loaders_round2
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


def train_epoch(model, makeup_loader, laptop_loader, corpus_loader, optimizer, scheduler=None):
	model.train()
	cum_loss = 0
	cum_lm_loss = 0
	total_lm_sample = 0
	P, G, S = 0, 0, 0
	total_sample = 0
	step = 0
	pbar = tqdm(range(max(len(makeup_loader), len(laptop_loader), len(corpus_loader))))
	makeup_iter = iter(makeup_loader)
	laptop_iter = iter(laptop_loader)
	corpus_iter = iter(corpus_loader)
	for _ in pbar:
		if step == max(len(makeup_loader), len(laptop_loader), len(corpus_loader)):
			pbar.close()
			break

		try:
			corpus_ids, corpus_attn, lm_label = next(corpus_iter)
		except StopIteration:
			corpus_iter = iter(corpus_loader)
			corpus_ids, corpus_attn, lm_label = next(corpus_iter)

		corpus_ids = corpus_ids.cuda()
		corpus_attn = corpus_attn.cuda()
		lm_label = lm_label.cuda()
		loss = model.foward_LM(corpus_ids, corpus_attn, lm_label)
		optimizer.zero_grad()
		loss.backward()
		if scheduler:
			scheduler.step()
		optimizer.step()
		cum_lm_loss += loss.data.cpu().numpy() * len(corpus_ids)
		total_lm_sample += len(corpus_ids)
		del corpus_ids, corpus_attn, lm_label, loss

		try:
			makeup_raw, makeup_x, makeup_y = next(makeup_iter)
		except StopIteration:
			makeup_iter = iter(makeup_loader)
			makeup_raw, makeup_x, makeup_y = next(makeup_iter)
		try:
			laptop_raw, laptop_x, laptop_y = next(laptop_iter)
		except StopIteration:
			laptop_iter = iter(laptop_loader)
			laptop_raw, laptop_x, laptop_y = next(laptop_iter)

		makeup_rv_raw, makeup_lb_raw = makeup_raw
		makeup_x = [item.cuda() for item in makeup_x]
		makeup_y = [item.cuda() for item in makeup_y]

		laptop_rv_raw, laptop_lb_raw = laptop_raw
		laptop_x = [item.cuda() for item in laptop_x]
		laptop_y = [item.cuda() for item in laptop_y]

		makeup_probs, makeup_logits = model.forward(makeup_x, type='makeup')
		makeup_loss = model.loss(makeup_logits, makeup_y) * len(makeup_rv_raw)

		laptop_probs, laptop_logits = model.forward(laptop_x, type='laptop')
		laptop_loss = model.loss(laptop_logits, laptop_y) * len(laptop_rv_raw)

		loss = (makeup_loss + laptop_loss) / (len(makeup_rv_raw) + len(laptop_rv_raw))
		optimizer.zero_grad()
		loss.backward()
		if scheduler:
			scheduler.step()
		optimizer.step()

		makeup_pred = model.gen_candidates(makeup_probs)
		makeup_pred = model.nms_filter(makeup_pred, 0.1)

		for b in range(len(makeup_pred)):
			gt = makeup_lb_raw[b]
			pred = [x[0] for x in makeup_pred[b]]
			p, g, s = evaluate_sample(gt, pred)
			P += p
			G += g
			S += s

		laptop_pred = model.gen_candidates(laptop_probs)
		laptop_pred = model.nms_filter(laptop_pred, 0.1)
		for b in range(len(laptop_pred)):
			gt = laptop_lb_raw[b]
			pred = [x[0] for x in laptop_pred[b]]
			p, g, s = evaluate_sample(gt, pred)
			P += p
			G += g
			S += s

		cum_loss += loss.data.cpu().numpy() * (len(makeup_rv_raw) + len(laptop_rv_raw))
		total_sample += (len(makeup_rv_raw) + len(laptop_rv_raw))
		step += 1
		while makeup_x:
			a = makeup_x.pop(); del a
			a = laptop_x.pop(); del a
		while makeup_y:
			a = makeup_y.pop(); del a
			a = laptop_y.pop(); del a

		while makeup_probs:
			a = makeup_probs.pop(); del a
			a = makeup_logits.pop(); del a
			a = laptop_probs.pop(); del a
			a = laptop_logits.pop(); del a

		del loss, makeup_loss, laptop_loss

	total_f1, total_pr, total_rc = f1_score(P, G, S)
	total_loss = cum_loss / total_sample
	total_lm_loss = cum_lm_loss / total_lm_sample
	return total_loss, total_lm_loss, total_f1, total_pr, total_rc


def eval_epoch(model, dataloader, type='makeup'):
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
			probs, logits = model.forward(x, type)
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
	tokenizer = BertTokenizer.from_pretrained('/home/zydq/.torch/models/bert/chinese_roberta_wwm_ext_L-12_H-768_A-12',
											  do_lower_case=True)
	# tokenizer = BertTokenizer.from_pretrained('/home/zydq/.tf/bert/chinese_roberta_wwm_ext_L-12_H-768_A-12',
	# 										  do_lower_case=True)

	makeup_train_loader, makeup_val_loader, laptop_train_loader, laptop_val_loader, corpus_loader = \
	get_data_loaders_round2(tokenizer, batch_size=12)


	model = OpinioNet.from_pretrained('/home/zydq/.torch/models/bert/chinese_roberta_wwm_ext_L-12_H-768_A-12')
	# model = OpinioNet.from_pretrained('/home/zydq/.tf/bert/chinese_roberta_wwm_ext_L-12_H-768_A-12', from_tf=True)
	model.cuda()
	optimizer = Adam(model.parameters(), lr=6e-6)
	scheduler = GradualWarmupScheduler(optimizer, total_epoch=2*max(len(makeup_train_loader), len(corpus_loader)))
	best_val_f1 = 0
	best_val_loss = float('inf')
	for e in range(EP):

		print('Epoch [%d/%d] train:' % (e, EP))
		train_loss, train_lm_loss, train_f1, train_pr, train_rc = train_epoch(model, makeup_train_loader, laptop_train_loader, corpus_loader, optimizer, scheduler)
		print("loss %.5f, lm loss %.5f f1 %.5f, pr %.5f, rc %.5f" % (train_loss, train_lm_loss, train_f1, train_pr, train_rc))

		print('Epoch [%d/%d] makeup eval:' % (e, EP))
		val_loss, val_f1, val_pr, val_rc = eval_epoch(model, makeup_val_loader, type='makeup')
		print("makeup_val: loss %.5f, f1 %.5f, pr %.5f, rc %.5f" % (val_loss, val_f1, val_pr, val_rc))

		print('Epoch [%d/%d] laptop eval:' % (e, EP))
		val_loss, val_f1, val_pr, val_rc = eval_epoch(model, laptop_val_loader, type='laptop')
		print("laptop_val: loss %.5f, f1 %.5f, pr %.5f, rc %.5f" % (val_loss, val_f1, val_pr, val_rc))

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