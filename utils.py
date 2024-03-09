
import os

import torch
import pandas as pd
import numpy as np
# from umap import UMAP
import umap.umap_ as UMAP
from anycache import anycache
import requests
from functools import wraps
from time import sleep


def initialize_model(model_class, layer_size, filepath):
    print("[-]\tLoading {} model...".format(filepath))
    if not os.path.exists(filepath):
        raise FileNotFoundError("{} is not found!".format(filepath))
    model = model_class(layer_size)
    model.load_state_dict(torch.load(
        filepath, map_location=torch.device('cpu')))
    model.eval()  # disable batch normalization & dropout (we're doing inference here)
    return model


@anycache(cachedir='.cache/')
def load_dataset(path):
    print("[-]\tLoading {} dataset...".format(path))
    df = pd.read_csv(path)
    data_summary = dict(df.iloc[:, 0].value_counts())
    X = []
    for i in range(df.shape[0]):
        X.append(list(map(int, list(df.iloc[i, 2]))))
    X = torch.tensor(np.array(X)).float()
    y = df.iloc[:, 0]
    return X, y, data_summary


@anycache(cachedir='.cache/')
def umap_transform(dae_obj, data_X, label_y):
    _, outputs = dae_obj(data_X)
    umap_data_obj = pd.DataFrame(
        UMAP(n_components=2).fit_transform(outputs.detach().numpy()))
    umap_data_obj['Label'] = label_y
    umap_data_obj = umap_data_obj.values.tolist()
    scatter_data = {}
    for x, y, data_label in umap_data_obj:
        if data_label not in scatter_data:
            scatter_data[data_label] = []
        scatter_data[data_label].append({'x': x, 'y': y})
    return scatter_data


@anycache(cachedir='.cache/')
def load_unigram(unigram_path):
    top_unigrams_var = []
    with open(unigram_path, 'r') as file_read:
        for each in file_read.read().split('\n'):
            if each:
                unigram, count = each.split('\t')
                top_unigrams_var.append({'Unigram': unigram, "Count": count})
    return top_unigrams_var


def fetch_malware_data(academic_api, filehash):
    params = {'apikey': academic_api, 'resource': filehash, 'allinfo': 1}
    response = requests.get(
        'https://www.virustotal.com/vtapi/v2/file/report', params=params)
    if response.status_code == 403:
        raise ValueError("VirusTotalError: API key returned status code 403!")
    json_data = response.json()
    if json_data['response_code'] == 0:
        raise ValueError("VirusTotalError: {}".format(json_data['verbose_msg']))
    if 'additional_info' not in json_data: 
        raise ValueError('VirusTotalError: {} contains no `additional_info` field!'.format(filehash))
    return json_data


def retry_call(exception_class, tries=50, delay=2, backoff=1.5, logger=None):

    def log_msg(msg):
        if logger:
            logger.warning(msg)
        else:
            print(msg)

    def deco_retry(f):

        @wraps(f)
        def f_retry(*args, **kwargs):
            local_delay = delay

            for i in range(tries):
                try:
                    data = f(*args, **kwargs)
                    return data
                except KeyboardInterrupt:
                    raise
                except exception_class as e:
                    if i == tries:
                        log_msg('Attempt failed')
                        return []
                    msg = '{}, Retrying in {} seconds...(trying {}/{})'.format(str(e), local_delay, i+1, tries)
                    log_msg(msg)
                    sleep(local_delay)
                    local_delay = round(local_delay * backoff, 2)
            else:
                return []  # max reties fail returning empty array, not raising error
            # return f(*args, **kwargs)

        return f_retry  # true decorator
    return deco_retry
