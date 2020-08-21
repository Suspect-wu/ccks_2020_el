

import json
from utils import kb_path, train_dev_path, dict_path, train_path, dev_path
from tqdm import tqdm

from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding


id2kb = {}
with open(kb_path, 'r', encoding='utf-8') as f:
    for i in tqdm(f):
        _ = json.loads(i)

        subject_id = _['subject_id']
        subject_type = _['type']

        subject_alias = list(set([_['subject']] + _.get('alias', [])))
        subject_alias = [i.lower() for i in subject_alias]

        _['data'].append({'predicate':'type', 'object' : subject_type})
        _['data'].append({
            'predicate' : u'名称',
            'object' : u'、'.join(subject_alias)
        })


        object_regex = set([i['predicate'] + ':' +i['object'] for i in _['data'] ])
        object_regex = sorted(object_regex, key=lambda s: len(s))
        object_regex = '|'.join(object_regex)
        object_regex.lower()


        id2kb[subject_id] = {
            # 别称
            'subject_alias' : subject_alias,
            # 特征
            'object_regex' : object_regex
        }
# for i in id2kb:
    # print(id2kb[i])
print(id2kb['169941'])

#%%

train_data = []
with open(dev_path, 'r', encoding='utf-8')as f:
    for i in tqdm(f):
        _ = json.loads(i)
        train_data.append({
            'text': _['text'],
            'mention_data': [
                (x['mention'].lower(), int(x['offset']), x['kb_id'])
                for x in _['mention_data'] if x['kb_id'] != 'NIL'
            ]
        })
print(train_data[-1])

#%%

kb2id = {}
for i, j in id2kb.items():
    for k in j['subject_alias']:
        if k not in kb2id:
            kb2id[k] = []
        kb2id[k].append(i)

print(kb2id['失踪女人'])



tokenizer = Tokenizer(token_dict=dict_path, do_lower_case=True)
# indice, segment = tokenizer.encode(first_text='你好', second_text='我难过', maxlen=250)
# [101, 872, 1962, 102, 2769, 7410, 6814, 102]
# [0, 0, 0, 0, 1, 1, 1, 1]
# print(indice)
# print(segment)
max_len = 0
all_len = []
max_a = 0
max_b = 0
inputs = {'ids':[], 'seg':[], 'entity_id':[], 'labels':[]}
for i in tqdm(train_data):
    text = i['text']
    for j in i['mention_data']:
        mention_name = j[0]
        if 'NIL' in j[2]:
            continue
        begin = int(j[1]) + 1
        end = begin + len(mention_name) - 1
        mention_kb_id = kb2id[mention_name]
        mention_kb_id = [i for i in mention_kb_id if i != j[2]]
        mention_kb_id = [j[2]] + mention_kb_id
        count = 0
        for k in mention_kb_id:
            if count > 2:
                break
            if count == 0:
                inputs['labels'].append(1)
            else:
                inputs['labels'].append(0)
            count += 1
            kb_text = id2kb[k]['object_regex']
            cur_text = text + kb_text
            cur_len = len(cur_text)
            # max_a = max(len(text), max_a)
            # max_b = max(len(kb_text), max_b)
            all_len.append(cur_len)
            indice, segment = tokenizer.encode(first_text=text, second_text=kb_text, maxlen=512)

            # print(len(indice), indice, segment)
            inputs['ids'].append(indice)
            inputs['seg'].append(segment)
            inputs['entity_id'].append([begin, end])
            # max_len = max(max_len, len(text + kb_text))

inputs['ids'] = sequence_padding(inputs['ids'])
inputs['seg'] = sequence_padding(inputs['seg'])

# print(max_len)
# print(max_a)
# print(max_b)
# import matplotlib.pyplot as plt
# plt.hist(all_len)
# plt.show()
# print(len(all_len))




#
#
#
#
#
# #%%
#
import numpy as np
import pandas as pd

#%%

for k in tqdm(inputs):
    inputs[k] = np.array(inputs[k])
    print(k, inputs[k].shape)
    print(inputs[k][1])
pd.to_pickle(inputs, r'./ccks2020_el_data/dev_iput.pkl')

#%%
import pandas as pd
data = pd.read_pickle('./ccks2020_el_data/dev_iput.pkl')
inputs = data['labels']
print(len(inputs))
print(data)


