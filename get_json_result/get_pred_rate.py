import json
from let_me_think import bert_model
from tqdm import tqdm
import numpy as np


input_file = './ccks2020_el_data/test_input.json'
output_file = './ccks2020_el_data/pred_output_1.json'

def get_input(input_file):
    data = input_file
    return [np.array(data['ids']), np.array(data['seg']), np.array(data['entity_id'])]

def predict_f1(input_file, out_file):
    result_list = []
    with open(input_file, 'r') as f:
        for line in f:
            temDict = json.loads(line)
            re_dict = {'text_id':temDict['text_id'], 'text':temDict['text']}
            re_dict['mention_data'] = []
            mention_data = temDict['mention_data']
            for men in mention_data:
                men.pop('link_data')
                men['link_pred'] = [0] * len(men['link_id'])
                re_dict['mention_data'].append(men)
            result_list.append(re_dict)


    model = bert_model(config, ckpt, pred=True)

    model.load_weights('./model/ED_binary_model_bert_f1.h5_new')

    with open(input_file, 'r')as f:
        for j, line in tqdm(enumerate(f)):
            temDict = json.loads(line)
            mention_data = temDict['mention_data']
            # 第J个句子对应的data
            re_men_data = result_list[j]['mention_data']
            for men, re_men in zip(mention_data, re_men_data):
                if len(men['link_id']) > 0:
                    pred = model.predict(get_input(men['link_data']))
                    re_men['link_pred'] = list(np.sum([re_men['link_pred'], list(np.squeeze(pred, axis=-1))], axis=0))
                    print(re_men['link_pred'])
                else:
                    re_men['link_pred'] = []


    out_file = open(out_file, 'w')
    for r in result_list:
        out_file.write(json.dumps(r, ensure_ascii=False))
        out_file.write('\n')


if __name__ == "__main__":
    config = 'roberta_zh_l12/bert_config.json'
    ckpt = 'roberta_zh_l12/bert_model.ckpt'
    predict_f1(input_file, output_file)