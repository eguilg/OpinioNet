import os
from pytorch_pretrained_bert import BertTokenizer
from dataset import get_pretrain_loaders
from model import OpinioNet

import torch
from torch.optim import Adam

from lr_scheduler import GradualWarmupScheduler, ReduceLROnPlateau
from tqdm import tqdm
import os.path as osp
import numpy as np
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
  # print(p, g, s)
  return p, g, s


def train_epoch(model, makeup_loader, corpus_loader, optimizer, scheduler=None):
  model.train()
  cum_loss = 0
  cum_lm_loss = 0
  total_lm_sample = 0
  P, G, S = 0, 0, 0
  total_sample = 0
  step = 0
  pbar = tqdm(range(max(len(makeup_loader), len(corpus_loader))))
  makeup_iter = iter(makeup_loader)
  corpus_iter = iter(corpus_loader)
  for _ in pbar:
    if step == max(len(makeup_loader), len(corpus_loader)):
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
    optimizer.step()
    if scheduler:
      scheduler.step()
    cum_lm_loss += loss.data.cpu().numpy() * len(corpus_ids)
    total_lm_sample += len(corpus_ids)
    del corpus_ids, corpus_attn, lm_label, loss

    try:
      makeup_raw, makeup_x, makeup_y = next(makeup_iter)
    except StopIteration:
      makeup_iter = iter(makeup_loader)
      makeup_raw, makeup_x, makeup_y = next(makeup_iter)

    makeup_rv_raw, makeup_lb_raw = makeup_raw
    makeup_x = [item.cuda() for item in makeup_x]
    makeup_y = [item.cuda() for item in makeup_y]

    makeup_probs, makeup_logits = model.forward(makeup_x, type='makeup')
    loss = model.loss(makeup_logits, makeup_y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if scheduler:
      scheduler.step()

    makeup_pred = model.gen_candidates(makeup_probs)
    makeup_pred = model.nms_filter(makeup_pred, 0.1)

    for b in range(len(makeup_pred)):
      gt = makeup_lb_raw[b]
      pred = [x[0] for x in makeup_pred[b]]
      p, g, s = evaluate_sample(gt, pred)
      P += p
      G += g
      S += s

    cum_loss += loss.data.cpu().numpy() * len(makeup_rv_raw)
    total_sample += len(makeup_rv_raw)
    step += 1
    while makeup_x:
      a = makeup_x.pop();
      del a
    while makeup_y:
      a = makeup_y.pop();
      del a

    while makeup_probs:
      a = makeup_probs.pop();
      del a
      a = makeup_logits.pop();
      del a

    del loss

  total_f1, total_pr, total_rc = f1_score(P, G, S)
  total_loss = cum_loss / total_sample
  total_lm_loss = cum_lm_loss / total_lm_sample
  return total_loss, total_lm_loss, total_f1, total_pr, total_rc


def eval_epoch(model, dataloader, type='makeup'):
  model.eval()
  cum_loss = 0
  # P, G, S = 0, 0, 0
  total_sample = 0
  step = 0
  pbar = tqdm(dataloader)

  PRED = []
  GT = []
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
    PRED += pred_result
    GT += lb_raw
    cum_loss += loss.data.cpu().numpy() * len(rv_raw)
    total_sample += len(rv_raw)

    step += 1

  total_loss = cum_loss / total_sample

  threshs = list(np.arange(0.1, 0.9, 0.05))
  best_f1, best_pr, best_rc = 0, 0, 0
  best_thresh = 0.1
  for th in threshs:
    P, G, S = 0, 0, 0
    PRED_COPY = copy.deepcopy(PRED)
    PRED_COPY = model.nms_filter(PRED_COPY, th)
    for b in range(len(PRED_COPY)):
      gt = GT[b]
      pred = [x[0] for x in PRED_COPY[b]]
      p, g, s = evaluate_sample(gt, pred)

      P += p
      G += g
      S += s
    f1, pr, rc = f1_score(P, G, S)
    if f1 > best_f1:
      best_f1, best_pr, best_rc = f1, pr, rc
      best_thresh = th

  return total_loss, best_f1, best_pr, best_rc, best_thresh


import argparse
from config import PRETRAINED_MODELS

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--base_model', type=str, default='roberta')
  parser.add_argument('--bs', type=int, default=12)
  parser.add_argument('--gpu', type=int, default=0)
  args = parser.parse_args()

  os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.gpu

  EP = 25
  model_config = PRETRAINED_MODELS[args.base_model]
  SAVING_DIR = '../models/'

  tokenizer = BertTokenizer.from_pretrained(model_config['path'], do_lower_case=True)
  makeup_train_loader, makeup_val_loader, corpus_loader = get_pretrain_loaders(tokenizer, batch_size=args.bs)
  model = OpinioNet.from_pretrained(model_config['path'], version=model_config['version'])
  model.cuda()
  optimizer = Adam(model.parameters(), lr=model_config['lr'])
  scheduler = GradualWarmupScheduler(optimizer, total_epoch=2 * max(len(makeup_train_loader), len(corpus_loader)))
  best_val_f1 = 0
  best_val_loss = float('inf')
  for e in range(EP):

    print('Epoch [%d/%d] train:' % (e, EP))
    train_loss, train_lm_loss, train_f1, train_pr, train_rc = train_epoch(model, makeup_train_loader, corpus_loader,
                                                                          optimizer, scheduler)
    print(
      "loss %.5f, lm loss %.5f f1 %.5f, pr %.5f, rc %.5f" % (train_loss, train_lm_loss, train_f1, train_pr, train_rc))

    print('Epoch [%d/%d] makeup eval:' % (e, EP))
    val_loss, val_f1, val_pr, val_rc, best_th = eval_epoch(model, makeup_val_loader, type='makeup')
    print("makeup_val: loss %.5f, f1 %.5f, pr %.5f, rc %.5f, thresh %.2f" % (val_loss, val_f1, val_pr, val_rc, best_th))

    if val_loss < best_val_loss:
      best_val_loss = val_loss
    if val_f1 > best_val_f1:
      best_val_f1 = val_f1
      if best_val_f1 >= 0.75:
        saving_dir = osp.join(SAVING_DIR, 'pretrained_' + model_config['name'])
        torch.save(model.state_dict(), saving_dir)
        print('saved best model to %s' % saving_dir)

    print('best loss %.5f' % best_val_loss)
    print('best f1 %.5f' % best_val_f1)
    if best_val_f1 >= 0.82:
      break
