import numpy as np
import synonyms
import pandas as pd

def is_intersec(s1, e1, s2, e2):
	if min(e1, e2) > max(s1, s2):
		return True
	return False

def aug_single():
	pass

def aug_df(reviews_df, labels_df, op, n=3):
	for idx in reviews_df.index:
		id = reviews_df.loc[idx, 'id']
		rv = reviews_df.loc[idx, 'Reviews']
		for i in reversed(range(len(rv))):
			if rv[i].strip() == '':
				for j in labels_df[labels_df['id'] == id].index:
					lb = labels_df[labels_df['id'] == id].loc[j]
					a_s = lb['A_start'].strip()
					a_e = lb['A_end'].strip()
					if a_s != '' and a_e != '':
						a_s = int(a_s)
						a_e = int(a_e)
						if a_s > i:
							a_s -= 1
							a_e -= 1
							labels_df.loc[j, 'A_start'] = str(a_s)
							labels_df.loc[j, 'A_end'] = str(a_e)
					o_s = lb['O_start'].strip()
					o_e = lb['O_end'].strip()
					if o_s != '' and o_e != '':
						o_s = int(o_s)
						o_e = int(o_e)
						if o_s > i:
							o_s -= 1
							o_e -= 1
							labels_df.loc[j, 'O_start'] = str(o_s)
							labels_df.loc[j, 'O_end'] = str(o_e)

		rv = rv.replace(' ', '')

		still_spans = []
		for i in labels_df[labels_df['id'] == id].index:
			lb = labels_df.loc[i]
			a_s = lb['A_start'].strip()
			a_e = lb['A_end'].strip()
			if a_s != '' and a_e != '':
				still_spans.append((int(a_s), int(a_e)))
			o_s = lb['O_start'].strip()
			o_e = lb['O_end'].strip()
			if o_s != '' and o_e != '':
				still_spans.append((int(o_s), int(o_e)))

		still_spans.sort(key=lambda x: x[0])

		rv_tokens = synonyms.seg(rv)[0]
		editable_tokens = []
		editable_spans = []
		cur = 0
		for i in range(len(rv_tokens)):

			end = cur + len(rv_tokens[i])
			editable = True
			for span in still_spans:
				if is_intersec(cur, end, span[0], span[1]):
					editable = False
					break
			if editable and (rv_tokens[i] not in ['，', ',', '！', '。', '*', '？', '?']):
				editable_spans.append((cur, end))
				editable_tokens.append(rv_tokens[i])
			cur = end

		if not editable_tokens:
			continue

		rv_list = list(rv)
		if op == 'delete' or op == 'replace' or op == 'insert':
			to_edit = sorted(np.random.choice(range(len(editable_tokens)), size=min(len(editable_tokens), n), replace=False),
				reverse=True)
			for ii in to_edit:
				span = editable_spans[ii]
				token = editable_tokens[ii]
				if op == 'delete' or op == 'replace':
					left, right = span
					if op == 'delete':
						target_token = ''
					else:
						candi, probs = synonyms.nearby(token)
						if len(candi) <= 1:
							target_token = ''
						else:
							probs = np.array(probs[1:]) / sum(probs[1:])
							target_token = np.random.choice(candi[1:], p=probs)
				else:
					left, right = span[-1], span[-1]
					token = ''
					candi, probs = synonyms.nearby(editable_tokens[ii])
					if len(candi) <= 1:
						target_token = ''
					else:
						probs = np.array(probs[1:]) / sum(probs[1:])
						target_token = np.random.choice(candi[1:], p=probs)

				shift = len(target_token)-len(token)

				for i in labels_df[labels_df['id'] == id].index:
					lb = labels_df.loc[i]
					a_s = lb['A_start'].strip()
					a_e = lb['A_end'].strip()
					if a_s != '' and a_e != '':
						a_s = int(a_s)
						a_e = int(a_e)
						if a_s >= span[-1]:
							a_s += shift
							a_e += shift
							labels_df.loc[i, 'A_start'] = str(a_s)
							labels_df.loc[i, 'A_end'] = str(a_e)
					o_s = lb['O_start'].strip()
					o_e = lb['O_end'].strip()
					if o_s != '' and o_e != '':
						o_s = int(o_s)
						o_e = int(o_e)
						if o_s >= span[-1]:
							o_s += shift
							o_e += shift
							labels_df.loc[i, 'O_start'] = str(o_s)
							labels_df.loc[i, 'O_end'] = str(o_e)
				print(token)
				print(''.join(rv_list[:left]), ''.join(rv_list[right:]))
				rv_list = rv_list[:left] + list(target_token) + rv_list[right:]

		elif op == 'swap':
			cur_time = 0
			if len(editable_tokens) < 2:
				continue
			if len(editable_tokens) == 2:
				time = 1
			else:
				time = n
			while cur_time != time:
				idx0, idx1 = sorted(np.random.choice(range(len(editable_tokens)), size=2, replace=False))
				token0, token1 = editable_tokens[idx0], editable_tokens[idx1]
				span0, span1 = editable_spans[idx0], editable_spans[idx1]
				print(token0, token1)
				editable_tokens[idx0], editable_tokens[idx1] = token1, token0
				if len(token0) != len(token1):
					shift = len(token1) - len(token0)
					editable_spans[idx0] = (span0[0], span0[0]+len(token1))
					editable_spans[idx1] = (span1[0]+shift, span1[0] + shift + len(token0))

					for idx_edt in range(len(editable_tokens)):
						cur_span = editable_spans[idx_edt]
						if cur_span[0] >= span0[1] and cur_span[1] <= span1[0]:
							editable_spans[idx_edt] = (cur_span[0]+shift, cur_span[1]+shift)

					for i in labels_df[labels_df['id'] == id].index:
						lb = labels_df.loc[i]
						a_s = lb['A_start'].strip()
						a_e = lb['A_end'].strip()
						if a_s != '' and a_e != '':
							a_s = int(a_s)
							a_e = int(a_e)
							if a_s >= span0[1] and a_e <= span1[0]:
								a_s += shift
								a_e += shift
								labels_df.loc[i, 'A_start'] = str(a_s)
								labels_df.loc[i, 'A_end'] = str(a_e)
						o_s = lb['O_start'].strip()
						o_e = lb['O_end'].strip()
						if o_s != '' and o_e != '':
							o_s = int(o_s)
							o_e = int(o_e)
							if o_s >= span0[1] and o_e <= span1[0]:
								o_s += shift
								o_e += shift
								labels_df.loc[i, 'O_start'] = str(o_s)
								labels_df.loc[i, 'O_end'] = str(o_e)

				rv_list = rv_list[:span0[0]] + list(token1) + rv_list[span0[1]: span1[0]] + list(token0) + rv_list[span1[1]:]

				cur_time += 1

		rv_new = ''.join(rv_list)
		reviews_df.loc[idx, 'Reviews'] = rv_new
		print(rv)
		print(rv_new)
		print(labels_df[labels_df['id'] == id])

	return reviews_df, labels_df

if __name__ == '__main__':
	reviews_df = pd.read_csv('../data/TRAIN/Train_laptop_reviews.csv', encoding='utf-8')
	labels_df = pd.read_csv('../data/TRAIN/Train_laptop_labels.csv', encoding='utf-8')

	reviews_df_replace, labels_df_replace = reviews_df.copy(), labels_df.copy()
	reviews_df_replace, labels_df_replace = aug_df(reviews_df_replace, labels_df_replace, 'replace')
	reviews_df_replace['id'] += reviews_df.shape[0]
	labels_df_replace['id'] += reviews_df.shape[0]

	reviews_df_insert, labels_df_insert = reviews_df.copy(), labels_df.copy()
	reviews_df_insert, labels_df_insert = aug_df(reviews_df_insert, labels_df_insert, 'insert')
	reviews_df_insert['id'] += reviews_df.shape[0]*2
	labels_df_insert['id'] += reviews_df.shape[0] * 2

	reviews_df_swap, labels_df_swap = reviews_df.copy(), labels_df.copy()
	reviews_df_swap, labels_df_swap = aug_df(reviews_df_swap, labels_df_swap, 'swap', 3)
	reviews_df_swap['id'] += reviews_df.shape[0]*3
	labels_df_swap['id'] += reviews_df.shape[0] * 3

	reviews_df_aug = pd.concat([reviews_df, reviews_df_replace, reviews_df_insert, reviews_df_swap], axis=0, ignore_index=True)
	labels_df_aug = pd.concat([labels_df, labels_df_replace, labels_df_insert, labels_df_swap], axis=0, ignore_index=True)

	reviews_df_aug.to_csv('../data/TRAIN/Train_laptop_aug_reviews.csv', index=False)
	labels_df_aug.to_csv('../data/TRAIN/Train_laptop_aug_labels.csv', index=False)