import json
from tqdm import tqdm
from bert4keras.tokenizers import Tokenizer

from utils import dict_path
import json
from let_try_mutil_nil_type import nil_type_model
import numpy as np
import tensorflow as tf
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
type2id = {'Event': 0, 'Person': 1, 'Work': 2, 'Location': 3, 'Time&Calendar': 4, 'Brand': 5, 'Natural&Geography': 6, 'Game': 7, 'Biological': 8, 'Medicine': 9, 'Food': 10, 'Software': 11, 'Vehicle': 12, 'Website': 13, 'Disease&Symptom': 14, 'Organization': 15, 'Awards': 16, 'Education': 17, 'Culture': 18, 'Constellation': 19, 'Law&Regulation': 20, 'VirtualThings': 21, 'Diagnosis&Treatment': 22, 'Other': 23}

id2type = {0: 'Event', 1: 'Person', 2: 'Work', 3: 'Location', 4: 'Time&Calendar', 5: 'Brand', 6: 'Natural&Geography', 7: 'Game', 8: 'Biological', 9: 'Medicine', 10: 'Food', 11: 'Software', 12: 'Vehicle', 13: 'Website', 14: 'Disease&Symptom', 15: 'Organization', 16: 'Awards', 17: 'Education', 18: 'Culture', 19: 'Constellation', 20: 'Law&Regulation', 21: 'VirtualThings', 22: 'Diagnosis&Treatment', 23: 'Other'}

prev_result = './ccks2020_el_data/result.json'

nil_path = './model/nil_loss.h5_multi_1'
model = nil_type_model()
model.load_weights(nil_path)

tokenizer = Tokenizer(do_lower_case=True, token_dict=dict_path)

finial_path = r'./ccks2020_el_data/finial_result_4.json'
result_list = []
with open(prev_result, 'r', encoding='utf-8')as f:
    for i in tqdm(f):
        _ = json.loads(i)
        mention_data = _['mention_data']
        text = _['text']
        for j in mention_data:
            if j['kb_id'] == 'NIL':
                begin = int(j['offset']) + 1
                end = begin + len(j['mention']) - 1
                indice, segment = tokenizer.encode(text, maxlen=46)
                indice += [0] * (46-len(indice))
                segment += [0] * (46-len(segment))
                rate = model.predict([np.array([indice]), np.array([segment]), np.array([[begin, end]])])
                rate = tf.reshape(shape=(-1,), tensor=rate)

                type_index = []
                for index, one_rate in enumerate(rate):
                    if one_rate > 0:

                        type_index.append(index)
                if len(type_index) == 1:
                    j['kb_id'] += '_' + id2type[type_index[0]]
                elif len(type_index) > 1:
                    type_str = ''
                    for i in type_index:
                        type_str = 'NIL_' + id2type[i] + '|'
                    type_str = type_str[:-1]
                    j['kb_id'] = type_str
                else:
                    j['kb_id'] = 'NIL_' + id2type[np.argmax(rate)]
                    print(rate[np.argmax(rate)])

                print(j['mention'], j['kb_id'])
        result_list.append(_)

with open(finial_path, 'w', encoding='utf-8')as r:
    for _ in result_list:
        r.write(json.dumps(_, ensure_ascii=False))
        r.write('\n')








