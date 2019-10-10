"""
Author： 周树帆 - SJTU
Email: sfzhou567@163.com
"""
import pandas as pd
from collections import defaultdict
import random
import numpy as np
from tqdm import tqdm


def data_augment(reviews_df, labels_df, epochs=5):
  POLARITY_DICT = 'polarity_dict'
  cate_dict = dict()
  for index, row in labels_df.iterrows():
    cate = row['Categories']
    aspect = row['AspectTerms']
    opinion, polarity = row['OpinionTerms'], row['Polarities']

    if cate not in cate_dict:
      cate_dict[cate] = {POLARITY_DICT: dict()}
    if polarity not in cate_dict[cate][POLARITY_DICT]:
      cate_dict[cate][POLARITY_DICT][polarity] = set()
    cate_dict[cate][POLARITY_DICT][polarity].add((aspect, opinion))

  global_review_id = 1
  new_reviews_df = pd.DataFrame(columns=reviews_df.columns)
  global_label_idx = 1
  new_labels_df = pd.DataFrame(columns=labels_df.columns)

  label_groups = labels_df.groupby('id')
  for epoch in range(epochs):
    for id, group in tqdm(label_groups):
      review = reviews_df.loc[id - 1]['Reviews']
      # TODO: 确认一下是否存在重叠的区间？ 然后把重叠的那部分都去掉

      ## region 区分极性，汇总数据, 然后遍历候选个数做aug
      polar_dict = defaultdict(list)
      for idx, row in group.iterrows():
        polarity = row['Polarities']
        polar_dict[polarity].append(idx)
      ## endregion
      for polar in polar_dict:
        indices = polar_dict[polar]
        for size in range(1, len(indices) + 1):
          new_group = group.copy()
          new_group['AspectOffset'] = 0
          new_group['OpinionOffset'] = 0
          chosen_indices = np.random.choice(indices, size, replace=False)
          for index in chosen_indices:
            row = new_group.loc[index]
            cate = row['Categories']
            aspect = row['AspectTerms']
            opinion, polarity = row['OpinionTerms'], row['Polarities']

            pair_set = cate_dict[cate][POLARITY_DICT][polarity]
            pair_list = list(pair_set)
            if len(pair_list) > 1:
              new_aspect, new_opinion = aspect, opinion
              accident_cnt = 0
              while (aspect == '_' and new_aspect != '_') or (new_aspect == aspect and new_opinion == opinion):
                new_idx = random.randint(0, len(pair_list) - 1)
                new_aspect, new_opinion = pair_list[new_idx]
                accident_cnt += 1
                if accident_cnt >= 1000:  # FIXME: 给原aspect为'_'的建一个dict来提速。不过现在这个代码也能用。
                  break
              new_group.loc[index, 'AspectTerms'] = new_aspect
              new_group.loc[index, 'AspectOffset'] = (0 if new_aspect == '_' else len(new_aspect)) - len(aspect)
              new_group.loc[index, 'OpinionTerms'] = new_opinion
              new_group.loc[index, 'OpinionOffset'] = (0 if new_opinion == '_' else len(new_opinion)) - len(opinion)

          ## 把spans拿出来排序，然后再塞回label里去
          spans = []
          span_set = set()
          for i, row in new_group.iterrows():
            aspect_element = {'idx': i,
                              'text': row['AspectTerms'],
                              'span': (row['A_start'], row['A_end']),
                              'offset': row['AspectOffset'],
                              'type': 'a'}
            if aspect_element['span'][0].strip() != '' and aspect_element['span'] not in span_set:
              span_set.add(aspect_element['span'])
              spans.append(aspect_element)
            opinion_element = {'idx': i,
                               'text': row['OpinionTerms'],
                               'span': (row['O_start'], row['O_end']),
                               'offset': row['OpinionOffset'],
                               'type': 'o'}
            if opinion_element['span'][0].strip() != '' and opinion_element['span'] not in span_set:
              span_set.add(opinion_element['span'])
              spans.append(opinion_element)
          sorted_spans = sorted(spans, key=lambda d: int(d['span'][0]))
          new_review = ''
          last_start = 0
          offset = 0
          for span in sorted_spans:
            # 下面3行顺序不能换, 必须是start, offset, end
            idx = span['idx']
            start = int(span['span'][0]) + offset if span['text'] != '_' else ' '
            offset += span['offset']
            end = int(span['span'][1]) + offset if span['text'] != '_' else ' '

            new_review += review[last_start:int(span['span'][0])] + (span['text'] if span['text'] != '_' else '')
            last_start = int(span['span'][1])

            if span['type'] == 'a':
              new_group.loc[idx, 'A_start'] = str(start)
              new_group.loc[idx, 'A_end'] = str(end)
            else:
              new_group.loc[idx, 'O_start'] = str(start)
              new_group.loc[idx, 'O_end'] = str(end)
          new_review += review[last_start:]

          ## 记录结果
          del new_group['AspectOffset']
          del new_group['OpinionOffset']
          for i, row in new_group.iterrows():
            row_data = row.tolist()
            row_data[0] = global_review_id
            new_labels_df.loc[global_label_idx] = row_data
            global_label_idx += 1
          new_reviews_df.loc[global_review_id] = [global_review_id, new_review]
          global_review_id += 1

  return new_reviews_df, new_labels_df


if __name__ == '__main__':
  data_type = 'laptop_corpus'
  epochs = 3  # 控制aug的倍数, 建议取<=5的正整数

  reviews_df = pd.read_csv('../data/TRAIN/Train_%s_reviews.csv' % data_type, encoding='utf-8')
  labels_df = pd.read_csv('../data/TRAIN/Train_%s_labels.csv' % data_type, encoding='utf-8')

  new_reviews_df, new_labels_df = data_augment(reviews_df, labels_df, epochs)

  new_reviews_df.to_csv('../data/TRAIN/Train_%s_aug_reviews.csv' % data_type, index=False, encoding='utf-8')
  new_labels_df.to_csv('../data/TRAIN/Train_%s_aug_labels.csv' % data_type, index=False, encoding='utf-8')
