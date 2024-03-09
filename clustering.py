from os import path
import glob
import re
import json
import random
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pickle
from kmeans import get_clustering


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


# Defining our function
def kmeans(x, k, no_of_iterations):
    idx = np.random.choice(len(x), k, replace=False)
    # Randomly choosing Centroids
    centroids = x[idx, :]  # Step 1

    # finding the distance between centroids and all the data points
    distances = cdist(x, centroids, 'euclidean')  # Step 2

    # Centroid with the minimum Distance
    points = np.array([np.argmin(i) for i in distances])  # Step 3

    # Repeating the above steps for a defined number of iterations
    # Step 4
    for _ in range(no_of_iterations):
        centroids = []
        for idx in range(k):
            # Updating Centroids by taking mean of Cluster it belongs to
            temp_cent = x[points == idx].mean(axis=0)
            centroids.append(temp_cent)

        centroids = np.vstack(centroids)  # Updated Centroids

        distances = cdist(x, centroids, 'euclidean')
        points = np.array([np.argmin(i) for i in distances])

    return points


def clustering(data_path='data/json', new_file_list=None):
    if new_file_list is None:
        return []
    unigrams_num = 1000
    if not path.exists(data_path):
        print("No data directory: {}".format(data_path))
        return []

    dict_unigram = dict()
    json_unigrams_list = []
    behaviors_json = list(glob.glob("{}/*.json".format(data_path)))
    new_file_index = [behaviors_json.index(path.join(data_path, x)) for x in new_file_list]

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

    pickle.dump(bit_string_list, open('data/data.pkl', 'wb'))

    print("\n[+] Calculating cluster numbers ...\n")
    cluster_numbers, k_model = get_clustering(bit_string_list)
    print("\nCluster number: {}".format(cluster_numbers))

    pickle.dump(k_model, open('data/model.pkl', 'wb'))
    print("\nSample number: {}".format(len(k_model.labels_)))
    print("Cluster number: {}".format(k_model.n_clusters))

    # get result
    result = []
    for i, index in enumerate(new_file_index):
        filename = new_file_list[i]
        class_label = k_model.labels_[index]
        result.append([filename, 'class-{}'.format(class_label)])
        print('filename: {}\tcluster: {}'.format(filename, class_label))

    # Visualize the results
    pca = PCA(2)
    df = pca.fit_transform(np.array(bit_string_list))
    label = kmeans(df, 10, 1000)
    u_labels = np.unique(k_model.labels_)
    for i in u_labels:
        plt.scatter(df[label == i, 0], df[label == i, 1], label=i)
    plt.legend()
    plt.show()
    plt.savefig('visualize.png')
    return result


if __name__ == '__main__':
    file_list = [
        'random_0a2e942bcd4bf41b40e44c1658348ba4.json',
        'random_fc49fa647a5f62d37dd3ca6f52ad1c53.json',
        'random_0b374fa6ffeff5261fcc12af8a23646a.json',
        'Cerber_00d96dcfe98e2b8c1374f8a8839561eaa33993909a8fa20289bb56d3b1bd41be.json'
    ]
    clustering(new_file_list=file_list)
