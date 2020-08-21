#%%
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding
import json
from utils import kb_path, train_dev_path
from tqdm import tqdm

#%%
all_type = ['Event','Person','Work','Location','Time&Calendar','Brand',
            'Natural&Geography','Game','Biological','Medicine','Food',
            'Software','Vehicle','Website','Disease&Symptom','Organization',
            'Awards','Education','Culture','Constellation','Law&Regulation',
            'VirtualThings','Diagnosis&Treatment','Other']
#%%

type2id = {}
count = 0
for type in all_type:
    if type not in type2id:
        type2id[type] = count
        count += 1
print(type2id)

id2type = {}

for item, value in type2id.items():
    id2type[value] = item
print(id2type)
print(len(id2type))


def get_type_label_existed(input):
    label = [0.] * 24
    if '|' in input:
        all_type = input.split('|')
        for in_type in all_type:
            label[type2id[in_type[:]]] = 1.
    else:
        label[type2id[input[:]]] = 1.
    return label

# kd_id : type
in_id2type = {}
typeset = {}
with open(kb_path, 'r', encoding='utf-8') as f:
    for i in tqdm(f):
        _ = json.loads(i)
        subject_tyep = _['type']
        subject_id = _['subject_id']
        in_id2type[subject_id] = get_type_label_existed(subject_tyep)
        typeset[subject_tyep] = typeset.get(subject_tyep, 0) + 1
print(in_id2type['10001'])

#%%

print(typeset)

#%%
# count = 0
# train_dev_type = set()
# with open(train_dev_path, 'r', encoding='utf-8')as f:
#     for i in tqdm(f):
#         _ = json.loads(i)
#         for j in _['mention_data']:
#             if 'NIL' in j['kb_id']:
#                 train_dev_type.add(j['kb_id'])
#                 count += 1
# print(train_dev_type)
# print(count)
# print(len(train_dev_type))
# for i in all_type:
#     if 'NIL_' + i not in train_dev_type:
#         print(i)
# for i in type_set:
#     if '|'not in i:
#         if 'NIL_' + i not in train_dev_type:
#             print(i)
#     else:
#         x = i.split('|')
#         for j, ii in enumerate(x):
#             other = 'NIL_' + ii
#             x[j] = other
#         if '|'.join(x) not in train_dev_type:
#             print('|'.join(x))

#
def get_type_label(input):
    label = [0.] * 24
    if '|' in input:
        all_type = input.split('|')
        for in_type in all_type:
            label[type2id[in_type[4:]]] = 1
    else:
        label[type2id[input[4:]]] = 1
    return label



train_type = []
with open(train_dev_path, 'r', encoding='utf-8')as f:
    for i in tqdm(f):
        _ = json.loads(i)
        train_type.append({
        'text':_['text'],
        'mention_data':[
            (int(x['offset']), get_type_label(x['kb_id']), x['mention'].lower())
            for x in _['mention_data'] if 'NIL' in x['kb_id']
                ]
                })
print(train_type[0])
print(len(train_type))


with open(train_dev_path, 'r', encoding='utf-8')as f:
    for i in tqdm(f):
        _ = json.loads(i)
        train_type.append({
        'text':_['text'],
        'mention_data':[
            (int(x['offset']), in_id2type[x['kb_id']], x['mention'].lower())
            for x in _['mention_data'] if 'NIL' not in x['kb_id']
                ]
                })
print(len(train_type))

#
from utils import dict_path
tokenizer = Tokenizer(do_lower_case=True, token_dict=dict_path)

#%%
#



#
# #%%
#
maxlen = 0
inputs = {'ids':[], 'seg':[], 'entity_id':[], 'label':[]}
for i in tqdm(train_type):
    text = i['text']
    maxlen = max(len(text), maxlen)
    mention_data = i['mention_data']
    indice, segment = tokenizer.encode(text, maxlen=55)
    for j in mention_data:
        mention = j[2]
        begin = int(j[0]) + 1
        end = begin + len(mention) - 1
        type = j[1]
        inputs['ids'].append(indice)
        inputs['label'].append(type)
        inputs['seg'].append(segment)
        inputs['entity_id'].append([begin, end])
print(maxlen)
inputs['ids'] = sequence_padding(inputs['ids'])
inputs['seg'] = sequence_padding(inputs['seg'])
#%%

import numpy as np
import pandas as pd

for k in tqdm(inputs):
    inputs[k] = np.array(inputs[k])
print(inputs['ids'].shape)
print(inputs['seg'].shape)
pd.to_pickle(inputs, r'./ccks2020_el_data/multi_nil_train_input.pkl')

#%%
