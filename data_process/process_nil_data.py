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

# id2type = {}
# type_set = set()
# with open(kb_path, 'r', encoding='utf-8') as f:
#     for i in tqdm(f):
#         _ = json.loads(i)
#         subject_tyep = _['type']
#         subject_id = _['subject_id']
#         id2type[subject_id] = subject_tyep
#         type_set.add(subject_tyep)
# print(id2type['10001'])
#
# #%%
#
# print(type_set)
NIL_TYPE_SET = ['NIL_Awards', 'NIL_Vehicle', 'NIL_Natural&Geography', 'NIL_Organization|NIL_VirtualThings',
                'NIL_Location|NIL_Organization', 'NIL_Education', 'NIL_Culture', 'NIL_VirtualThings',
                'NIL_Game', 'NIL_Person|NIL_VirtualThings', 'NIL_Biological', 'NIL_Brand|NIL_Other',
                'NIL_Work|NIL_Education', 'NIL_Medicine', 'NIL_Disease&Symptom', 'NIL_VirtualThings|NIL_Person',
                'NIL_Location', 'NIL_Brand|NIL_Organization', 'NIL_Organization', 'NIL_Brand|NIL_Organization|NIL_Location',
                'NIL_Organization|NIL_Location', 'NIL_Brand|NIL_Location', 'NIL_Food', 'NIL_Work|NIL_Other',
                'NIL_Event|NIL_Work', 'NIL_Event|NIL_Education', 'NIL_Time&Calendar', 'NIL_Brand', 'NIL_Diagnosis&Treatment',
                'NIL_Law&Regulation|NIL_Work|NIL_Other', 'NIL_Work|NIL_Person', 'NIL_Website|NIL_Software',
                'NIL_Law&Regulation', 'NIL_Software|NIL_Website', 'NIL_Work', 'NIL_Event|NIL_Other', 'NIL_Other',
                'NIL_Website', 'NIL_Software', 'NIL_Person', 'NIL_Event', 'NIL_VirtualThings|NIL_Organization',
                'NIL_VirtualThings|NIL_Location', 'NIL_Time&Calendar|NIL_Other', 'NIL_Time&Calendar|NIL_Person',
                'NIL_Location|NIL_VirtualThings']

#%%

type2id = {}
count = 0
for type in NIL_TYPE_SET:
    if type not in type2id:
        type2id[type] = count
        count += 1
print(type2id)

id2type = {}

for item, value in type2id.items():
    id2type[value] = item
print(id2type)
print(len(id2type))

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
train_type = []
with open(train_dev_path, 'r', encoding='utf-8')as f:
    for i in tqdm(f):
        _ = json.loads(i)
        train_type.append({
        'text':_['text'],
        'mention_data':[
            (int(x['offset']), type2id[x['kb_id']], x['mention'].lower())
            for x in _['mention_data'] if 'NIL' in x['kb_id']
                ]
                })




print(train_type[0])

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

inputs['ids'] = sequence_padding(inputs['ids'])
inputs['seg'] = sequence_padding(inputs['seg'])
#%%

import numpy as np
import pandas as pd

for k in tqdm(inputs):
    inputs[k] = np.array(inputs[k])
print(inputs['ids'].shape)
print(inputs['seg'].shape)
pd.to_pickle(inputs, r'./ccks2020_el_data/nil_train_input.pkl')

#%%
