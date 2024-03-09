import os
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score
import pickle
import numpy as np


def get_clustering(src_data=None, is_file=False, debug_flag=False):
    if is_file:
        if os.path.isfile(src_data):
            features = np.array(pickle.load(open(src_data, 'rb')))
        else:
            print("There is no file: {}".format(src_data))
            return 0
    else:
        features = np.array(src_data)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # How to Choose the Number of Clusters
    # kmeans_kwargs = {
    #     "init": "random",
    #     "n_init": 10,
    #     "max_iter": 500,
    #     "n_jobs": -1,  # using all processors
    #     "random_state": 42,
    # }
    kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 500,
    "random_state": 42,
}


    kmodels = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit_predict(scaled_features)
        kmodels.append(kmeans)
        # sse.append(kmeans.inertia_)

    # A list holds the SSE values for each k
    sse = [x.inertia_ for x in kmodels]

    if debug_flag:
        plt.style.use("fivethirtyeight")
        plt.plot(scaled_features)
        plt.show()

    kl = KneeLocator(
        range(1, 11), sse, curve="convex", direction="decreasing"
    )
    cn = kl.elbow
    model = kmodels[cn-1]

    return cn, model
