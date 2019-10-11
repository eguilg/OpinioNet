import os
from pytorch_pretrained_bert import BertTokenizer
from dataset import ReviewDataset, get_data_loaders_cv, get_aug_data_loaders_cv
from lr_scheduler import GradualWarmupScheduler, ReduceLROnPlateau
from model import OpinioNet

import torch
from torch.optim import Adam

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
  return p, g, s


def train_epoch(model, dataloader, optimizer, scheduler=None, type='makeup'):
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

    probs, logits = model.forward(x, type)
    loss = model.loss(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if scheduler:
      scheduler.step()

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

  threshs = list(np.arange(0.1, 0.9, 0.025))
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


import json
import argparse
from config import PRETRAINED_MODELS

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--base_model', type=str, default='roberta')
  parser.add_argument('--bs', type=int, default=12)
  parser.add_argument('--gpu', type=int, default=0)
  args = parser.parse_args()

  # os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.gpu

  EP = 100
  FOLDS = 5
  SAVING_DIR = '../models/'
  THRESH_DIR = '../models/thresh_dict.json'
  model_config = PRETRAINED_MODELS[args.base_model]
  print(model_config)

  if osp.isfile(THRESH_DIR):
    with open(THRESH_DIR, 'r', encoding='utf-8') as f:
      thresh_dict = json.load(f)
  else:
    thresh_dict = {}

  tokenizer = BertTokenizer.from_pretrained(model_config['path'], do_lower_case=True)
  cv_loaders = get_data_loaders_cv(rv_path='../data/TRAIN/Train_laptop_reviews.csv',
                                   lb_path='../data/TRAIN/Train_laptop_labels.csv',
                                   tokenizer=tokenizer,
                                   batch_size=args.bs,
                                   type='laptop',
                                   folds=FOLDS)

  BEST_THRESHS = [0.1] * FOLDS
  BEST_F1 = [0] * FOLDS
  for cv_idx, (train_loader, val_loader) in enumerate(cv_loaders):
    model = OpinioNet.from_pretrained(model_config['path'], version=model_config['version'], focal=model_config['focal'])
    model.load_state_dict(torch.load('../models/pretrained_' + model_config['name']))
    model.cuda()
    optimizer = Adam(model.parameters(), lr=model_config['lr'])
    scheduler = GradualWarmupScheduler(optimizer, total_epoch=10 * len(train_loader))
    best_val_f1 = 0
    best_val_loss = float('inf')

    for e in range(EP):

      print('Epoch [%d/%d] train:' % (e, EP))
      train_loss, train_f1, train_pr, train_rc = train_epoch(model, train_loader, optimizer, scheduler, type='laptop')
      print("train: loss %.5f, f1 %.5f, pr %.5f, rc %.5f" % (train_loss, train_f1, train_pr, train_rc))

      print('Epoch [%d/%d] eval:' % (e, EP))
      val_loss, val_f1, val_pr, val_rc, best_th = eval_epoch(model, val_loader, type='laptop')
      print("val: loss %.5f, f1 %.5f, pr %.5f, rc %.5f, thresh %.2f" % (val_loss, val_f1, val_pr, val_rc, best_th))

      if val_loss < best_val_loss:
        best_val_loss = val_loss
      if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        if val_f1 >= 0.75:
          saving_name = model_config['name'] + '_cv' + str(cv_idx)
          saving_dir = osp.join(SAVING_DIR, saving_name)
          torch.save(model.state_dict(), saving_dir)
          print('saved best model to %s' % saving_dir)
          BEST_THRESHS[cv_idx] = best_th
          BEST_F1[cv_idx] = best_val_f1
          thresh_dict[saving_name] = {
            'name': model_config['name'],
            'thresh': best_th,
            'f1': best_val_f1,
          }
          with open(THRESH_DIR, 'w', encoding='utf-8') as f:
            json.dump(thresh_dict, f)

      print('best loss %.5f' % best_val_loss)
      print('best f1 %.5f' % best_val_f1)

    del model, optimizer, scheduler
  print(BEST_THRESHS)
  print(BEST_F1)
