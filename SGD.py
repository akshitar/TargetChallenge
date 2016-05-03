import csv
from sklearn.feature_extraction.text import CountVectorizer
import sklearn
import nltk
import pandas as pd
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
vec = CountVectorizer(analyzer='word')
#
userProducts = dict()
with open('data/transactionLog.csv', 'rb') as transactionLog:
    count = -1
    reader = csv.reader(transactionLog)
    for userData in reader:
        count = count + 1
        if (count == 0):
            continue
        if count % 1000 == 0:
            print(count)
            if count > 1000:
                break
        uID = str(userData[0])
        skuID = str(userData[1])

        # Append products to the user profile
        if uID in userProducts:
            userProducts[uID].append(skuID)
        else:
            userProducts[uID] = [skuID]

userProductList = userProducts.items()
index = []
tags = []

for x in userProductList:
    index.append(x[0])
    tags.append(' '.join(x[1]))
data = vec.fit_transform(tags).toarray()
print(data)
#
df = pd.DataFrame(data, columns = vec.get_feature_names())
df['index'] = index
df = df.set_index('index')
df = df.transpose()
print(df)
#
R = df.as_matrix()
numUsers = df.shape[1]
numItems = df.shape[0]
#
# #Initialize the parameters
# ################################# These needs to be varied
k = 500
lam = 50
iter = 5
eta = 0.01

# Random initialization of P and Q
P = random.rand(numUsers, k)
Q = random.rand(numItems, k)

# P = np.zeros((numUsers, k))#*sqrt(5.0/k)
# Q = np.zeros((numItems, k))#*sqrt(5.0/k)
obj = empty(iter)
print()
# Update P and Q
for i in range(0, iter):
    error=0
    for user in range(0,numUsers):
        for item in range(0,numItems):
            eps = (R[item, user]-dot(Q[item],P[user]))
            temp_Q = Q[item] + eta*(eps*P[user]-lam*Q[item])
            temp_P = P[user] + eta*(eps*Q[item]-lam*P[user])
            Q[item] = temp_Q
            P[user] = temp_P

            error += (R[item, user]-dot(Q[item],P[user]))**2
    # error = error + lam*(sum(P**2)+sum(Q**2))
    obj[i] = error
    print(i)
    print(error)


# Plot the value of objective function as a function of iteration
fig = plt.figure()
xvals = range(0,iter)
yvals = obj
plt.plot(xvals, yvals)
plt.xlabel("Iterations")
plt.ylabel("Objective function's values")
plt.title("Plot of Objcetive function vs Iterations")
plt.show()
