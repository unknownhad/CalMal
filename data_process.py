from os import path
import glob
import re
import json
import random
import pandas as pd
from config import config


def preprocess_json(json_vt_data):
    behaviors_data = json.dumps(json_vt_data['additional_info'])
    # split by newlines
    behaviors_data = behaviors_data.replace('\n', ' ')
    # remove many spaces into single space
    behaviors_data = re.sub(r'\s+', ' ', behaviors_data)
    # remove hash
    behaviors_data = re.sub(r'[a-fA-F\d]{32,128}', '', behaviors_data)
    behaviors_data = re.sub(r'\d+\.\d+', '', behaviors_data)
    # remove empty string
    unigrams = list(set(filter(None, behaviors_data.split())))
    r = re.compile(r'^"|",?:?$|,$')
    # remove some unuseful chars
    unigrams = [r.sub('', unigram) for unigram in unigrams if r.search(unigram)]
    # strip each unigrams
    unigrams = [unigram.strip() for unigram in unigrams]
    # remove unigram if length <= 3
    unigrams = [unigram for unigram in unigrams if len(unigram) > 3]
    return unigrams


def main():
    data_path = config.get_json_data_path
    if not path.exists(data_path):
        print("No data directory: {}".format(data_path))
        return

    unigrams_num = 10000
    dict_unigram = dict()
    json_unigrams_list = []
    behaviors_json = list(glob.glob("{}/*.json".format(data_path)))

    print("\n[+] Splitting unigrams for every JSON files ...\n")

    for i, each_json in enumerate(behaviors_json):
        with open(each_json, 'r') as json_fi:

            if i % 10 == 0:
                print("\33[2K\rProcessing [{}/{}] -- {}".format(i, len(behaviors_json), each_json), end='')

            unigrams = preprocess_json(json.loads(json_fi.read()))
            json_unigrams_list.append(unigrams)

            for unigram in unigrams:
                dict_unigram[unigram] = 1 if unigram not in dict_unigram else dict_unigram[unigram] + 1

    print("\33[2K\r[+] Choosing {} top unigrams from most frequent unigrams ...".format(unigrams_num))

    top_unigrams = []

    for key, val in dict_unigram.items():
        if val < len(behaviors_json)-10:  # if occurs to all files
            top_unigrams.append((key, val))
    top_unigrams.sort(reverse=True, key=lambda x: x[1])
    top_unigrams = top_unigrams[:unigrams_num]
    random.shuffle(top_unigrams)

    print("\n[+] Converting all files' unigrams into bit-string based on top-frequent unigrams ...\n")

    bit_string_list = []

    for i, json_unigram in enumerate(json_unigrams_list):
        if i % 10 == 0:
            print("\33[2K\rProcessing {}/{}".format(i, len(json_unigrams_list)), end='')
        json_unigram = set(json_unigram)  # for faster lookup O(n log n)
        bit_string = [int(frequent_unigram in json_unigram) for frequent_unigram, count in top_unigrams]
        bit_string_list.append(bit_string)

    print("\33[2K\r[+] Saving into dataset.tar.xz and top_unigrams.txt ...\n")

    dataset_list = []

    for filename, bit_string in zip(behaviors_json, bit_string_list):
        label, filehash = path.basename(filename).split('_')
        filehash = filehash.split('.')[0]
        dataset_list.append([label] + [filehash] + bit_string)

    with open(config.unigram_path, 'w') as top_unigrams_fo:
        top_unigrams_fo.write('\n'.join(['\t'.join(str(x) for x in each_unigram) for each_unigram in top_unigrams]))

    df = pd.DataFrame(dataset_list)
    df.to_csv(config.dataset_path, index=False, compression='xz')


if __name__ == '__main__':
    main()
