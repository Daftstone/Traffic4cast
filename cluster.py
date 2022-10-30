import nltk.cluster.util
import pandas as pd
import numpy as np
import os

# 城市
city = 'london'

# begin-end 聚类数量选择，这里就是从20个类到30个类中选择最优的聚类数
begin = 20
end = 20

paths = os.listdir("temp/")
sum_data = 0.
info_list = []

count = 0
for path in paths:
    if ("data_%s" % city in path):
        cur_path = os.path.join('temp/', path)
        cur_data = np.load(cur_path)

        tt = np.sum(cur_data, axis=1)
        ii = np.where((np.isnan(tt) == False))[0]
        if (count == 0):
            idx = set(list(ii))
        else:
            idx = idx & set(list(ii))

        info = path.strip("\n").split("_")
        info_list.append((info[2], int(info[3][:-4])))
        count += 1
idx = np.array(list(idx))
print(len(idx))

vector = np.zeros((len(info_list), len(idx)))

count = 0
for path in paths:
    if ("data_%s" % city in path):
        cur_path = os.path.join('temp/', path)
        cur_data = np.load(cur_path)
        means = np.mean(cur_data[idx], axis=1).reshape(-1)
        vector[count] = means
        count += 1

from sklearn.cluster import KMeans
from nltk.cluster.kmeans import KMeansClusterer
from nltk.cluster import euclidean_distance
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA, TruncatedSVD
from scipy.spatial import distance

X = vector
print(X.shape)

scores = []
for k in range(begin, end + 1):
    print(k)
    cluster = KMeans(n_clusters=k, random_state=0, max_iter=1000).fit(X)
    score = silhouette_score(X, cluster.labels_, metric='euclidean')
    scores.append(score)
k = np.argsort(-np.array(scores)) + begin
cluster = KMeans(n_clusters=k[0], random_state=0, max_iter=1000).fit(X)
y_pred = cluster.labels_

maps = {i: [] for i in range(k[0])}
for info, idx in zip(info_list, y_pred):
    maps[idx].append((info[0], info[1], 1))

for key, value in maps.items():
    print(key, len(value))

import pickle

with open("temp/%s.pkl" % city, "wb") as f:
    pickle.dump(maps, f)

idx = np.where(y_pred == 0)[0]
print(vector[idx][:10, :20])
print(np.mean(vector[idx] / 4), np.std(vector[idx] / 4))

idx = np.where(y_pred == 9)[0]
print(vector[idx][:10, :20])
print(np.mean(vector[idx] / 4), np.std(vector[idx] / 4))
