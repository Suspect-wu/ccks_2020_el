import json
import numpy as np

def result(input_file, output_file):
    output_file = open(output_file, 'w')
    count = 0
    with open(input_file, 'r')as f:
        for line in f:
            _ = json.loads(line)
            mention_data = _['mention_data']
            for men in mention_data:
                if len(men['link_id']) > 0:
                    link_pred = men['link_pred']
                    arg_max = int(np.argmax(link_pred))
                    if link_pred[arg_max] > 0.5:
                        men['kb_id'] = men['link_id'][arg_max]
                    else:
                        print(men['mention'])
                        men['kb_id'] = 'NIL'
                        count += 1
                else:
                    men['kb_id'] = 'NIL'
                men.pop('link_id')
                men.pop('link_pred')
            output_file.write(json.dumps(_, ensure_ascii=False))
            output_file.write('\n')
    print(count)

if __name__ == '__main__':
    input_file = './ccks2020_el_data/pred_output_1.json'
    output_file = './ccks2020_el_data/result_1.json'
    result(input_file, output_file)

