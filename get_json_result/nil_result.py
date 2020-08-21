import json
from tqdm import tqdm
from bert4keras.tokenizers import Tokenizer

from utils import dict_path
import json
from roberta_nil_type_model import nil_type_model
import numpy as np
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
type2id = {'NIL_Awards': 0, 'NIL_Vehicle': 1, 'NIL_Natural&Geography': 2, 'NIL_Organization|NIL_VirtualThings': 3, 'NIL_Location|NIL_Organization': 4, 'NIL_Education': 5, 'NIL_Culture': 6, 'NIL_VirtualThings': 7, 'NIL_Game': 8, 'NIL_Person|NIL_VirtualThings': 9, 'NIL_Biological': 10, 'NIL_Brand|NIL_Other': 11, 'NIL_Work|NIL_Education': 12, 'NIL_Medicine': 13, 'NIL_Disease&Symptom': 14, 'NIL_VirtualThings|NIL_Person': 15, 'NIL_Location': 16, 'NIL_Brand|NIL_Organization': 17, 'NIL_Organization': 18, 'NIL_Brand|NIL_Organization|NIL_Location': 19, 'NIL_Organization|NIL_Location': 20, 'NIL_Brand|NIL_Location': 21, 'NIL_Food': 22, 'NIL_Work|NIL_Other': 23, 'NIL_Event|NIL_Work': 24, 'NIL_Event|NIL_Education': 25, 'NIL_Time&Calendar': 26, 'NIL_Brand': 27, 'NIL_Diagnosis&Treatment': 28, 'NIL_Law&Regulation|NIL_Work|NIL_Other': 29, 'NIL_Work|NIL_Person': 30, 'NIL_Website|NIL_Software': 31, 'NIL_Law&Regulation': 32, 'NIL_Software|NIL_Website': 33, 'NIL_Work': 34, 'NIL_Event|NIL_Other': 35, 'NIL_Other': 36, 'NIL_Website': 37, 'NIL_Software': 38, 'NIL_Person': 39, 'NIL_Event': 40, 'NIL_VirtualThings|NIL_Organization': 41, 'NIL_VirtualThings|NIL_Location': 42, 'NIL_Time&Calendar|NIL_Other': 43, 'NIL_Time&Calendar|NIL_Person': 44, 'NIL_Location|NIL_VirtualThings': 45}
id2type = {0: 'NIL_Awards', 1: 'NIL_Vehicle', 2: 'NIL_Natural&Geography', 3: 'NIL_Organization|NIL_VirtualThings', 4: 'NIL_Location|NIL_Organization', 5: 'NIL_Education', 6: 'NIL_Culture', 7: 'NIL_VirtualThings', 8: 'NIL_Game', 9: 'NIL_Person|NIL_VirtualThings', 10: 'NIL_Biological', 11: 'NIL_Brand|NIL_Other', 12: 'NIL_Work|NIL_Education', 13: 'NIL_Medicine', 14: 'NIL_Disease&Symptom', 15: 'NIL_VirtualThings|NIL_Person', 16: 'NIL_Location', 17: 'NIL_Brand|NIL_Organization', 18: 'NIL_Organization', 19: 'NIL_Brand|NIL_Organization|NIL_Location', 20: 'NIL_Organization|NIL_Location', 21: 'NIL_Brand|NIL_Location', 22: 'NIL_Food', 23: 'NIL_Work|NIL_Other', 24: 'NIL_Event|NIL_Work', 25: 'NIL_Event|NIL_Education', 26: 'NIL_Time&Calendar', 27: 'NIL_Brand', 28: 'NIL_Diagnosis&Treatment', 29: 'NIL_Law&Regulation|NIL_Work|NIL_Other', 30: 'NIL_Work|NIL_Person', 31: 'NIL_Website|NIL_Software', 32: 'NIL_Law&Regulation', 33: 'NIL_Software|NIL_Website', 34: 'NIL_Work', 35: 'NIL_Event|NIL_Other', 36: 'NIL_Other', 37: 'NIL_Website', 38: 'NIL_Software', 39: 'NIL_Person', 40: 'NIL_Event', 41: 'NIL_VirtualThings|NIL_Organization', 42: 'NIL_VirtualThings|NIL_Location', 43: 'NIL_Time&Calendar|NIL_Other', 44: 'NIL_Time&Calendar|NIL_Person', 45: 'NIL_Location|NIL_VirtualThings'}

prev_result = './ccks2020_el_data/result.json'

nil_path = './model/nil_loss.h5_other'
model = nil_type_model()
model.load_weights(nil_path)

tokenizer = Tokenizer(do_lower_case=True, token_dict=dict_path)

finial_path = r'./ccks2020_el_data/finial_result_1.json'
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
                max_index = np.argmax(rate.reshape(-1,))
                j['kb_id'] = id2type[max_index]
                print(j['mention'], j['kb_id'])
        with open(finial_path, 'a+', encoding='utf-8')as r:
            r.write(json.dumps(_, ensure_ascii=False))
            r.write('\n')








