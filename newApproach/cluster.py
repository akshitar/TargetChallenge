import csv
import marshal as pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
tfidf_vectorizer = CountVectorizer(analyzer='word')

def cleanArr(arr):
    for i in range(0, len(arr)):
        arr[i] = arr[i].replace('token_', '')
        arr[i] = arr[i].strip()
    return arr

fileUserFeatures = open('../assets/userFeatures', 'r+')
fileUserProducts = open('../assets/userProducts', 'r+')
fileUserClusters = open('../assets/userClusters', 'r+')

productFeatures = {}
userFeatures = {}
userProducts = {}

count = 0
with open('../data/skuFeatures.csv', 'rb') as features:
    reader = csv.reader(features)
    for row in reader:
        if count != 0:
            productFeatures[row[0]] = cleanArr(row[1].split(' '))
        count = count + 1
print('Found all the product features')

count = 0
with open('../data/transactionLog.csv', 'rb') as transactionLog:
    reader = csv.reader(transactionLog)
    next(reader)
    for userData in reader:
        uID = str(userData[0])
        skuID = str(userData[1])
        count = count + 1
        if count % 10000000 == 0:
            print(count)
            if count > 30000000:
                break
        if skuID in productFeatures:
            # Append product features to the user profile
            if uID in userFeatures:
                userFeatures[uID] = userFeatures[uID] + productFeatures[skuID]
            else:
                userFeatures[uID] = productFeatures[skuID]
print('Found all user features and user products')

try:
    # pickle.dump(userProducts, fileUserProducts)
    # userProducts = pickle.load(fileUserProducts)
    # print('Load user products')
    #
    # pickle.dump(userFeatures, fileUserFeatures)
    # userFeatures = pickle.load(fileUserFeatures)
    print('Load user features')
except Exception as e:
    print(e)
    print('Load failed')

features = []
index = []
for user in userFeatures:
    features.append(' '.join(list(set(userFeatures[user]))))
    index.append(user)

print('Calculating count vector')
tfidf_matrix = tfidf_vectorizer.fit_transform(features).toarray()
df = pd.DataFrame(tfidf_matrix, columns = tfidf_vectorizer.get_feature_names())

print('Clustering')
num_clusters = 3
km = KMeans(n_clusters=num_clusters)
km.fit(df)
clusters = km.labels_.tolist()

clusterGroups = np.zeros([num_clusters, 0]).tolist()
for x in range(0, len(clusters)):
    clusterGroups[clusters[x]].append(index[x])

print('Saving clusters')
pickle.dump(clusterGroups, fileUserClusters)
print('Done')
