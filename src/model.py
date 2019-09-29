import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertOnlyMLMHead
from pytorch_pretrained_bert import BertModel, BertAdam, BertConfig

from dataset import ID2P, ID2COMMON, ID2MAKUP, ID2LAPTOP
import numpy as np

from collections import Counter

def focalBCE_with_logits(logits, target, gamma=2):
	probs = torch.sigmoid(logits)
	grad = torch.abs(target - probs) ** gamma
	grad /= grad.mean()
	loss = grad * F.binary_cross_entropy(probs, target, reduction='none')
	return loss.mean()


class OpinioNet(BertPreTrainedModel):
	def __init__(self, config, hidden=100, gpu=True, dropout_prob=0.3, bert_cache_dir=None):
		super(OpinioNet, self).__init__(config)

		self.bert_cache_dir = bert_cache_dir

		self.bert = BertModel(config)
		self.apply(self.init_bert_weights)
		self.bert_hidden_size = self.config.hidden_size

		self.w_as11 = nn.Linear(self.bert_hidden_size, hidden)
		self.w_as12 = nn.Linear(self.bert_hidden_size, hidden)
		self.w_ae11 = nn.Linear(self.bert_hidden_size, hidden)
		self.w_ae12 = nn.Linear(self.bert_hidden_size, hidden)
		self.w_os11 = nn.Linear(self.bert_hidden_size, hidden)
		self.w_os12 = nn.Linear(self.bert_hidden_size, hidden)
		self.w_oe11 = nn.Linear(self.bert_hidden_size, hidden)
		self.w_oe12 = nn.Linear(self.bert_hidden_size, hidden)

		self.w_as2 = nn.Linear(hidden, 1)
		self.w_ae2 = nn.Linear(hidden, 1)
		self.w_os2 = nn.Linear(hidden, 1)
		self.w_oe2 = nn.Linear(hidden, 1)

		self.w_obj = nn.Linear(self.bert_hidden_size, 1)

		self.w_common = nn.Linear(self.bert_hidden_size, len(ID2COMMON))
		self.w_makeup = nn.Linear(self.bert_hidden_size, len(ID2MAKUP) - len(ID2COMMON))
		self.w_laptop = nn.Linear(self.bert_hidden_size, len(ID2LAPTOP) - len(ID2COMMON))
		self.w_p = nn.Linear(self.bert_hidden_size, len(ID2P))


		self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)

		# self.w_num = nn.Linear(self.bert_hidden_size, 8)

		self.dropout = nn.Dropout(dropout_prob)

		self.softmax = nn.Softmax(dim=-1)
		self.log_softmax = nn.LogSoftmax(dim=-1)

		self.kl_loss = nn.KLDivLoss(reduction='batchmean')

		if gpu:
			self.cuda()

	def foward_LM(self, input_ids, attention_mask=None, masked_lm_labels=None):
		sequence_output, _ = self.bert(input_ids, None, attention_mask,
									   output_all_encoded_layers=False)
		prediction_scores = self.cls(sequence_output)

		if masked_lm_labels is not None:
			loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
			masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
			return masked_lm_loss
		else:
			return prediction_scores

	def forward(self, input, type='laptop'):
		rv_seq, att_mask, rv_mask = input

		rv_seq, cls_emb = self.bert(input_ids=rv_seq, attention_mask=att_mask, output_all_encoded_layers=False)
		# rv_seq = self.dropout(rv_seq)
		# cls_emb = self.dropout(cls_emb)

		as_logits = self.w_as2(F.leaky_relu(self.w_as11(self.dropout(rv_seq)).unsqueeze(2)
											+ self.w_as12(self.dropout(rv_seq)).unsqueeze(1))).squeeze(-1)

		ae_logits = self.w_ae2(F.leaky_relu(self.w_ae11(self.dropout(rv_seq)).unsqueeze(2)
											+ self.w_ae12(self.dropout(rv_seq)).unsqueeze(1))).squeeze(-1)

		os_logits = self.w_os2(F.leaky_relu(self.w_os11(self.dropout(rv_seq)).unsqueeze(2)
											+ self.w_os12(self.dropout(rv_seq)).unsqueeze(1))).squeeze(-1)

		oe_logits = self.w_oe2(F.leaky_relu(self.w_oe11(self.dropout(rv_seq)).unsqueeze(2)
											+ self.w_oe12(self.dropout(rv_seq)).unsqueeze(1))).squeeze(-1)

		obj_logits = self.w_obj(self.dropout(rv_seq)).squeeze(-1)

		# c_logits = self.w_c(self.dropout(rv_seq))
		common_logits = self.w_common(self.dropout(rv_seq))
		if type == 'laptop':
			special_logits = self.w_laptop(self.dropout(rv_seq))
		else:
			special_logits = self.w_makeup(self.dropout(rv_seq))

		c_logits = torch.cat([common_logits, special_logits], dim=-1)
		p_logits = self.w_p(self.dropout(rv_seq))

		# num_logits = self.w_num(self.dropout(cls_emb))

		rv_mask_with_cls = rv_mask.clone()
		rv_mask_with_cls[:, 0] = 1
		pointer_mask = rv_mask_with_cls.unsqueeze(2) * rv_mask_with_cls.unsqueeze(1)
		pointer_mask[:, 0, :] = 0

		pointer_mask = (1 - pointer_mask).byte()
		rv_mask = (1 - rv_mask).byte()

		as_logits = as_logits.masked_fill(pointer_mask, -1e5)
		ae_logits = ae_logits.masked_fill(pointer_mask, -1e5)
		os_logits = os_logits.masked_fill(pointer_mask, -1e5)
		oe_logits = oe_logits.masked_fill(pointer_mask, -1e5)

		obj_logits = obj_logits.masked_fill(rv_mask, -1e5)

		probs = [self.softmax(as_logits),
				 self.softmax(ae_logits),
				 self.softmax(os_logits),
				 self.softmax(oe_logits),
				 torch.sigmoid(obj_logits),
				 self.softmax(c_logits),
				 self.softmax(p_logits)]

		logits = [as_logits, ae_logits, os_logits, oe_logits, obj_logits, c_logits, p_logits]

		return probs, logits

	def loss(self, preds, targets):
		as_logits, ae_logits, os_logits, oe_logits, obj_logits, c_logits, p_logits = preds
		as_tgt, ae_tgt, os_tgt, oe_tgt, obj_tgt, c_tgt, p_tgt = targets

		as_logits = as_logits.permute((0, 2, 1))
		ae_logits = ae_logits.permute((0, 2, 1))
		os_logits = os_logits.permute((0, 2, 1))
		oe_logits = oe_logits.permute((0, 2, 1))
		c_logits = c_logits.permute((0, 2, 1))
		p_logits = p_logits.permute((0, 2, 1))

		loss = 0
		loss += F.cross_entropy(as_logits, as_tgt, ignore_index=-1)
		loss += F.cross_entropy(ae_logits, ae_tgt, ignore_index=-1)
		loss += F.cross_entropy(os_logits, os_tgt, ignore_index=-1)
		loss += F.cross_entropy(oe_logits, oe_tgt, ignore_index=-1)

		loss += F.binary_cross_entropy_with_logits(obj_logits, obj_tgt)
		# loss += focalBCE_with_logits(obj_logits, obj_tgt)

		loss += F.cross_entropy(c_logits, c_tgt, ignore_index=-1)
		loss += F.cross_entropy(p_logits, p_tgt, ignore_index=-1)

		# loss += F.cross_entropy(num_logits, num_tgt, ignore_index=-1)

		return loss

	def gen_candidates(self, probs, thresh=0.01):
		as_probs, ae_probs, os_probs, oe_probs, obj_probs, c_probs, p_probs = probs
		as_scores, as_preds = as_probs.max(dim=-1)
		ae_scores, ae_preds = ae_probs.max(dim=-1)
		os_scores, os_preds = os_probs.max(dim=-1)
		oe_scores, oe_preds = oe_probs.max(dim=-1)

		c_scores, c_preds = c_probs.max(dim=-1)
		p_scores, p_preds = p_probs.max(dim=-1)
		confidence = (
				as_scores * ae_scores * os_scores * oe_scores * p_scores * c_scores * obj_probs).data.cpu().numpy()

		as_preds = as_preds.data.cpu().numpy()
		ae_preds = ae_preds.data.cpu().numpy()
		os_preds = os_preds.data.cpu().numpy()
		oe_preds = oe_preds.data.cpu().numpy()

		conf_rank = (-confidence).argsort(-1)

		c_preds = c_preds.data.cpu().numpy()
		p_preds = p_preds.data.cpu().numpy()

		result = []
		for b in range(len(c_preds)):
			sample_res = []
			for pos in conf_rank[b]:
				if sample_res and confidence[b][pos] < thresh:
					break
				a_s = as_preds[b][pos]
				a_e = ae_preds[b][pos]
				o_s = os_preds[b][pos]
				o_e = oe_preds[b][pos]

				cls = c_preds[b][pos]
				polar = p_preds[b][pos]

				conf = confidence[b][pos]

				# 检查自身是否合理
				if a_s > a_e:
					continue
				if o_s > o_e:
					continue
				if min(a_e, o_e) >= max(a_s, o_s):  # 内部重叠
					continue

				# 检查与前面的是否重叠
				# is_bad = False
				# for sample in sample_res:
				# 	s1, e1, s2, e2 = sample[0][:4]
				# 	if min(a_e, e1) >= max(a_s, s1) and min(o_e, e2) >= max(o_s, s2):
				# 		is_bad = True
				# 		break
				# if is_bad:
				# 	continue

				sample_res.append(((a_s, a_e, o_s, o_e, cls, polar), conf))
			result.append(sample_res)
		return result

	def beam_search(self, probs, thresh=0.01):
		as_probs, ae_probs, os_probs, oe_probs, obj_probs, c_probs, p_probs = probs

		c_scores, c_preds = c_probs.max(dim=-1)
		p_scores, p_preds = p_probs.max(dim=-1)

		c_preds = c_preds.data.cpu().numpy()
		p_preds = p_preds.data.cpu().numpy()

		as_sorted = as_probs.argsort(dim=-1, descending=True)
		ae_sorted = ae_probs.argsort(dim=-1, descending=True)
		os_sorted = os_probs.argsort(dim=-1, descending=True)
		oe_sorted = oe_probs.argsort(dim=-1, descending=True)
		max_conf = (as_probs.gather(dim=2, index=as_sorted[:, :, 0:1]).squeeze(-1) *
					ae_probs.gather(dim=2, index=ae_sorted[:, :, 0:1]).squeeze(-1) *
					os_probs.gather(dim=2, index=os_sorted[:, :, 0:1]).squeeze(-1) *
					oe_probs.gather(dim=2, index=oe_sorted[:, :, 0:1]).squeeze(-1) *
					obj_probs * c_scores * p_scores)
		conf_rank = max_conf.argsort(dim=-1, descending=True)
		result = []

		for b in range(len(conf_rank)):
			sample_res = []
			# print('=====start====')
			for pos_idx in range(len(conf_rank[b])):

				pos = conf_rank[b][pos_idx]
				cur_conf = max_conf[b][pos].data.cpu().item()
				# print(max_conf[b])
				# print(as_probs[b][pos])
				# print('entering position %d, conf: %.5f' % (pos, cur_conf))
				if cur_conf < thresh and len(sample_res) > 0:
					break
				as_idx, ae_idx, os_idx, oe_idx = 0, 0, 0, 0

				cls = c_preds[b][pos]
				polar = p_preds[b][pos]

				while True:

					a_s = as_sorted[b][pos][as_idx].data.cpu().item()
					a_e = ae_sorted[b][pos][ae_idx].data.cpu().item()
					o_s = os_sorted[b][pos][os_idx].data.cpu().item()
					o_e = oe_sorted[b][pos][oe_idx].data.cpu().item()

					is_bad = False
					# 检查自身是否合理
					if a_s > a_e:
						is_bad = True
					elif o_s > o_e:
						is_bad = True
					elif min(a_e, o_e) >= max(a_s, o_s):  # 内部重叠
						is_bad = True

					# 继续搜索
					# print(a_s, a_e, o_s, o_e, cur_conf, is_bad)
					if is_bad:
						if as_idx != as_sorted.shape[-1] - 1:
							as_cur_score = as_probs[b][pos][as_sorted[b][pos][as_idx]].data.cpu().item()
							as_nxt_score = as_probs[b][pos][as_sorted[b][pos][as_idx + 1]].data.cpu().item()
							nxt_as_conf = cur_conf * as_nxt_score / as_cur_score
						else:
							nxt_as_conf = 0

						if ae_idx != ae_sorted.shape[-1] - 1:
							ae_cur_score = ae_probs[b][pos][ae_sorted[b][pos][ae_idx]].data.cpu().item()
							ae_nxt_score = ae_probs[b][pos][ae_sorted[b][pos][ae_idx + 1]].data.cpu().item()
							nxt_ae_conf = cur_conf * ae_nxt_score / ae_cur_score
						else:
							nxt_ae_conf = 0

						if os_idx != os_sorted.shape[-1] - 1:
							os_cur_score = os_probs[b][pos][os_sorted[b][pos][os_idx]].data.cpu().item()
							os_nxt_score = os_probs[b][pos][os_sorted[b][pos][os_idx + 1]].data.cpu().item()
							nxt_os_conf = cur_conf * os_nxt_score / os_cur_score
						else:
							nxt_os_conf = 0

						if oe_idx != oe_sorted.shape[-1] - 1:
							oe_cur_score = oe_probs[b][pos][oe_sorted[b][pos][oe_idx]].data.cpu().item()
							oe_nxt_score = oe_probs[b][pos][oe_sorted[b][pos][oe_idx + 1]].data.cpu().item()
							nxt_oe_conf = cur_conf * oe_nxt_score / oe_cur_score
						else:
							nxt_oe_conf = 0

						if nxt_as_conf == nxt_ae_conf == nxt_os_conf == nxt_oe_conf == 0:
							break

						max_conf_idx = np.argmax([nxt_as_conf, nxt_ae_conf, nxt_os_conf, nxt_oe_conf])
						if max_conf_idx == 0:
							as_idx += 1
							cur_conf = nxt_as_conf
						elif max_conf_idx == 1:
							ae_idx += 1
							cur_conf = nxt_ae_conf
						elif max_conf_idx == 2:
							os_idx += 1
							cur_conf = nxt_os_conf
						else:
							oe_idx += 1
							cur_conf = nxt_oe_conf
						# print('next conf', cur_conf)
						if cur_conf < thresh and len(sample_res) > 0:
							break

					# if pos_idx != len(conf_rank[b]) - 1:
					# 	nxt_pos = conf_rank[b][pos_idx + 1]
					# 	nxt_max_conf = max_conf[b][nxt_pos].data.cpu().item()
					# 	if cur_conf < nxt_max_conf:
					# 		break

					else:
						# print('inserting ', ((a_s, a_e, o_s, o_e, cls, polar), cur_conf))
						sample_res.append(((a_s, a_e, o_s, o_e, cls, polar), cur_conf))
						break

			result.append(sample_res)
		return result

	@staticmethod
	def nms_filter(results, thresh=0.1):
		for i, opinions in enumerate(results):
			# 对于重复结果分数取平均值
			# scores = {}
			# for k, v in opinions:
			# 	if k not in scores:
			# 		scores[k] = [0, 0] # 总分、数量
			# 	scores[k][0] += v
			# 	scores[k][1] += 1
			# for k in scores.keys():
			# 	scores[k] = scores[k][0] / scores[k][1]
			# # 按照分数排序 进入nms筛选
			# opinions = sorted(list(scores.items()), key=lambda x: -x[1])
			opinions = sorted(opinions, key=lambda x: -x[1])
			nmsopns = []
			for opn in opinions:
				if opn[1] < thresh and len(nmsopns) > 0:
					break
				isbad = False
				for nmsopn in nmsopns:
					as1, ae1, os1, oe1 = opn[0][:4]
					as2, ae2, os2, oe2 = nmsopn[0][:4]
					if min(ae1, ae2) >= max(as1, as2) and min(oe1, oe2) >= max(os1, os2):
						isbad = True
						break
				if not isbad:
					nmsopns.append(opn)
			results[i] = nmsopns
			# print(results)
		return results


if __name__ == '__main__':
	from pytorch_pretrained_bert import BertTokenizer
	from dataset import ReviewDataset

	tokenizer = BertTokenizer.from_pretrained('/home/zydq/.torch/models/bert/ERNIE',
											  do_lower_case=True)
	model = OpinioNet.from_pretrained('/home/zydq/.torch/models/bert/ERNIE')
	model.cuda()
	model.train()

	d = ReviewDataset('../data/TRAIN/Train_reviews.csv', '../data/TRAIN/Train_labels.csv', tokenizer)
	b_raw, b_in, b_tgt = d.batchify(d[:10])

	for i in range(len(b_in)):
		b_in[i] = b_in[i].cuda()
	for i in range(len(b_tgt)):
		b_tgt[i] = b_tgt[i].cuda()
	print(b_in)
	probs, logits = model.forward(b_in)
	loss = model.loss(logits, b_tgt)
	result = model.nms(probs)
	print(loss)
	print(result)
