#%%

import json
from utils import kb_path, dict_path, test_path
from tqdm import tqdm
import re
import numpy as np
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding


#%%
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

#%% dict

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

#%% 读取测试数据
def get_link_entity_test(mention):
    if mention in kb2id:
        return list(kb2id[mention])
    return []
#%%
all = []
out_file = './ccks2020_el_data/test_input.json'
with open(test_path, 'r', encoding='utf-8')as f:
    for i in tqdm(f):
        cur_dict = json.loads(i)
        text = cur_dict['text']
        mention_data = cur_dict['mention_data']
        for i, j in enumerate(mention_data):
            j['mention'].lower()
            mention = j['mention']
            offset = int(j['offset'])
            begin = int(offset) + 1
            end = begin + len(mention) - 1
            link_id = get_link_entity_test(mention)
            j['link_id'] = link_id  ··1·
            link_data = {'ids':[], 'seg':[], 'entity_id':[]}
            for id in link_id:
                kb_text = id2kb[id]['object_regex']
                cur_text = text + kb_text
                indice, segment = tokenizer.encode(first_text=text,
                                                   second_text=kb_text,
                                                   maxlen=512)

                link_data['ids'].append(indice)
                link_data['seg'].append(segment)
                link_data['entity_id'].append([begin, end])
            if link_data['ids']:
                link_data['ids'] = sequence_padding(link_data['ids'], length=512).tolist()
                link_data['seg'] = sequence_padding(link_data['seg'], length=512).tolist()
            j['link_data'] = link_data
        all.append(cur_dict)

with open(out_file, 'w', encoding='utf-8')as r:
    for i in all:
       r.write(json.dumps(i, ensure_ascii=False))
       r.write('\n')