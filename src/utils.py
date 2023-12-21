# -*- coding: utf-8 -*-


import logging
import random
import numpy as np
import ujson as json
import torch
try:
    from .plm_checkers.checker_utils import soft_logic
except:
    from plm_checkers.checker_utils import soft_logic


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def init_logger(level, filename=None, mode='a', encoding='utf-8'):
    logging_config = {
        'format': '%(asctime)s - %(levelname)s - %(name)s:\t%(message)s',
        'datefmt': '%Y-%m-%d %H:%M:%S',
        'level': level,
        'handlers': [logging.StreamHandler()]
    }
    if filename:
        logging_config['handlers'].append(logging.FileHandler(filename, mode, encoding))
    logging.basicConfig(**logging_config)


def read_json(filename, mode='r', encoding='utf-8'):
    with open(filename, mode, encoding=encoding) as fin:
        return json.load(fin)


def save_json(data, filename, mode='w', encoding='utf-8'):
    with open(filename, mode, encoding=encoding) as fout:
        json.dump(data, fout, ensure_ascii=False, indent=4)


def read_json_lines(filename, mode='r', encoding='utf-8', skip=0):
    with open(filename, mode, encoding=encoding) as fin:
        for line in fin:
            if skip > 0:
                skip -= 1
                continue
            yield json.loads(line)


def save_json_lines(data, filename, mode='w', encoding='utf-8', skip=0):
    with open(filename, mode, encoding=encoding) as fout:
        for line in data:
            if skip > 0:
                skip -= 1
                continue
            print(json.dumps(line, ensure_ascii=False), file=fout)


def read_json_dict(filename, mode='r', encoding='utf-8'):
    with open(filename, mode, encoding=encoding) as fin:
        key_2_id = json.load(fin)
        id_2_key = dict(zip(key_2_id.values(), key_2_id.keys()))

    return key_2_id, id_2_key


def save_json_dict(data, filename, mode='w', encoding='utf-8'):
    with open(filename, mode, encoding=encoding) as fout:
        json.dump(data, fout, ensure_ascii=False, indent=4)


# Calculate precision, recall and f1 value
# According to https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure
def get_prf(res):
    if res['TP'] == 0:
        if res['FP'] == 0 and res['FN'] == 0:
            precision = 1.0
            recall = 1.0
            f1 = 1.0
        else:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
    else:
        precision = 1.0 * res['TP'] / (res['TP'] + res['FP'])
        recall = 1.0 * res['TP'] / (res['TP'] + res['FN'])
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1

def compute_metrics(truth, predicted, z_predicted):
    assert len(truth) == len(predicted)

    outputs = []
    results = {}
    cnt = 0
    z_cnt_h, z_cnt_s = 0, 0  # acc_h,acc_s
    agree_h, agree_s = 0, 0  # AGREE
    y_zh = []
    y_zs = []
    y_zs_logit = []
    z_list = []
    for x, y, z in zip(truth, predicted, z_predicted):
        res = {'label': x, 'prediction': y}
        if x == y:
            cnt += 1

        res['pred_z'] = z


        y_ = soft_logic(torch.tensor([z]))[0]
        y_zs_logit.append(y_)

       
        if y_[0] > 0.3:
            ys_temp = 0
        else:
            ys_temp = 1


        y_zs.append(ys_temp)

        if ys_temp == x:
            z_cnt_s += 1
        if ys_temp == y:
            agree_s += 1



        z_111=[1, 1, 1]

        threshold = 0.5

        z_h = torch.ones(3)
        for i in range(3):
            if i == 0:
                if z[i][0] > threshold:
                    z_h[i] = 0
            elif i == 1:
                if z[i][0] > threshold:
                    z_h[i] = 0
            else:
                if z[i][0] > threshold:
                    z_h[i] = 0
        z_h = z_h.tolist()

        z_list.append(z_h)
        if 1 in z_h:  # FAKE
            if z_h == z_111:
                y__ = 2
            else:
                y__ = 1
        else:  # REAL
            y__ = 0
        y_zh.append(y__)
        if y__ == x:
            z_cnt_h += 1
        if y__ == y:
            agree_h += 1

        outputs.append(res)

    results['Accuracy'] = cnt / len(truth)
    results['z_Acc_hard'] = z_cnt_h / len(truth)
    results['z_Acc_soft'] = z_cnt_s / len(truth)
    results['Agreement_hard'] = agree_h / len(truth)
    results['Agreement_soft'] = agree_s / len(truth)
    return outputs, results, y_zh, z_list, y_zs, y_zs_logit
