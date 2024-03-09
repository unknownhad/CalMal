
import json
import re

import torch


def preprocess_json(json_vt_data):

    # only take additional_info data
    behaviors_data = json.dumps(json_vt_data['additional_info'])
    behaviors_data = behaviors_data.replace('\n', ' ')  # split by newlines
    # remove many spaces into single space
    behaviors_data = re.sub(r'\s+', ' ', behaviors_data)
    behaviors_data = re.sub(
        r'[a-fA-F\d]{32,128}', '', behaviors_data)  # remove hash
    behaviors_data = re.sub(r'\d+\.\d+', '', behaviors_data)
    # remove empty string
    unigrams = list(set(filter(None, behaviors_data.split())))
    # remove some chars
    r = re.compile(r'^"|",?:?$|,$')
    for i in range(len(unigrams)):
        if r.search(unigrams[i]):
            unigrams[i] = r.sub('', unigrams[i])
    # strip each unigrams
    unigrams = [each_unigram.strip() for each_unigram in unigrams]
    # remove unigram if length <= 3
    unigrams = [
        each_unigram for each_unigram in unigrams if len(each_unigram) > 3]

    return unigrams


def unigrams_to_bitstring(malware_unigrams_list, top_unigram_list):

    unigram_bitstr = ''
    for each_top_unigrams in top_unigram_list:
        unigram_bitstr += str(
            int(each_top_unigrams['Unigram'] in malware_unigrams_list))
    return unigram_bitstr


def gen_signs_from_bitstring(dae_obj, unigram_bitstr):
    unigram_bitstr = list(map(int, unigram_bitstr))
    X = torch.tensor(unigram_bitstr).view(-1, 10000).float()
    encoded, _ = dae_obj(X)
    return encoded.detach().numpy().tolist()


def predict_from_malware_sign(mlp_obj, mal_sign, label_index):

    X = torch.tensor(mal_sign)
    y = mlp_obj(X)
    probability = dict(zip(label_index, torch.nn.Softmax()(y.data)
                           .detach().numpy().tolist()[0]))
    _, pred_label = torch.max(y.data, 1)
    y_index = pred_label.detach().numpy().tolist()[0]
    return probability, label_index[y_index]
