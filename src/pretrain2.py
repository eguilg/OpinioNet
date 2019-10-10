import os
from pytorch_pretrained_bert import BertTokenizer
from dataset import get_pretrain2_loaders
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


def train_epoch(model, makeup_loader, laptop_loader, corpus_loader, optimizer, scheduler=None):
  model.train()

  cum_lm_loss = 0
  cum_makeup_loss = 0
  cum_laptop_loss = 0
  total_lm_sample = 0
  total_makeup_sample = 0
  total_laptop_sample = 0
  P_makeup, G_makeup, S_makeup = 0, 0, 0
  P_laptop, G_laptop, S_laptop = 0, 0, 0
  step = 0
  epoch_len = max(len(makeup_loader), len(corpus_loader), len(laptop_loader))
  pbar = tqdm(range(epoch_len))

  corpus_iter = iter(corpus_loader)
  makeup_iter = iter(makeup_loader)
  laptop_iter = iter(laptop_loader)

  for _ in pbar:
    if step == epoch_len:
      pbar.close()
      break
    ################ MLM ###################
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

    ############### makeup ##################
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
      P_makeup += p
      G_makeup += g
      S_makeup += s

    cum_makeup_loss += loss.data.cpu().numpy() * len(makeup_rv_raw)
    total_makeup_sample += len(makeup_rv_raw)
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

    ############### laptop ##################
    try:
      laptop_raw, laptop_x, laptop_y = next(laptop_iter)
    except StopIteration:
      laptop_iter = iter(laptop_loader)
      laptop_raw, laptop_x, laptop_y = next(laptop_iter)

    laptop_rv_raw, laptop_lb_raw = laptop_raw
    laptop_x = [item.cuda() for item in laptop_x]
    laptop_y = [item.cuda() for item in laptop_y]

    laptop_probs, laptop_logits = model.forward(laptop_x, type='laptop')
    loss = model.loss(laptop_logits, laptop_y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if scheduler:
      scheduler.step()

    laptop_pred = model.gen_candidates(laptop_probs)
    laptop_pred = model.nms_filter(laptop_pred, 0.1)

    for b in range(len(laptop_pred)):
      gt = laptop_lb_raw[b]
      pred = [x[0] for x in laptop_pred[b]]
      p, g, s = evaluate_sample(gt, pred)
      P_laptop += p
      G_laptop += g
      S_laptop += s

    cum_laptop_loss += loss.data.cpu().numpy() * len(laptop_rv_raw)
    total_laptop_sample += len(laptop_rv_raw)
    while laptop_x:
      a = laptop_x.pop();
      del a
    while laptop_y:
      a = laptop_y.pop();
      del a

    while laptop_probs:
      a = laptop_probs.pop();
      del a
      a = laptop_logits.pop();
      del a

    del loss

    step += 1

  total_lm_loss = cum_lm_loss / total_lm_sample

  makeup_f1, makeup_pr, makeup_rc = f1_score(P_makeup, G_makeup, S_makeup)
  makeup_loss = cum_makeup_loss / total_makeup_sample

  laptop_f1, laptop_pr, laptop_rc = f1_score(P_laptop, G_laptop, S_laptop)
  laptop_loss = cum_laptop_loss / total_laptop_sample

  return makeup_loss, makeup_f1, makeup_pr, makeup_rc, \
         laptop_loss, laptop_f1, laptop_pr, laptop_rc, \
         total_lm_loss


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
  makeup_loader, laptop_loader, laptop_val_loader, corpus_loader = get_pretrain2_loaders(tokenizer, batch_size=args.bs)
  model = OpinioNet.from_pretrained(model_config['path'], version=model_config['version'])
  model.cuda()
  optimizer = Adam(model.parameters(), lr=model_config['lr'])
  scheduler = GradualWarmupScheduler(optimizer,
                                     total_epoch=2 * max(len(makeup_loader), len(laptop_loader), len(corpus_loader)))
  best_val_f1 = 0
  best_val_loss = float('inf')
  for e in range(EP):

    print('Epoch [%d/%d] train:' % (e, EP))
    makeup_loss, makeup_f1, makeup_pr, makeup_rc, \
    laptop_loss, laptop_f1, laptop_pr, laptop_rc, \
    total_lm_loss = train_epoch(model, makeup_loader, laptop_loader, corpus_loader, optimizer, scheduler)
    print("makeup_train: loss %.5f, f1 %.5f, pr %.5f, rc %.5f" % (makeup_loss, makeup_f1, makeup_pr, makeup_rc))
    print("laptop_train: loss %.5f, f1 %.5f, pr %.5f, rc %.5f" % (laptop_loss, laptop_f1, laptop_pr, laptop_rc))
    print("lm loss %.5f", total_lm_loss)

    print('Epoch [%d/%d] makeup eval:' % (e, EP))
    val_loss, val_f1, val_pr, val_rc, best_th = eval_epoch(model, laptop_val_loader, type='laptop')
    print("makeup_val: loss %.5f, f1 %.5f, pr %.5f, rc %.5f, thresh %.2f" % (val_loss, val_f1, val_pr, val_rc, best_th))

    if val_loss < best_val_loss:
      best_val_loss = val_loss
    if val_f1 > best_val_f1:
      best_val_f1 = val_f1
      if best_val_f1 >= 0.75:
        saving_dir = osp.join(SAVING_DIR, 'pretrained2_' + model_config['name'])
        torch.save(model.state_dict(), saving_dir)
        print('saved best model to %s' % saving_dir)

    print('best loss %.5f' % best_val_loss)
    print('best f1 %.5f' % best_val_f1)
    if best_val_f1 >= 0.82:
      break
