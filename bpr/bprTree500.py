"""
Bayesian Personalized Ranking

Matrix Factorization model and a variety of classes
implementing different sampling strategies.
"""

import numpy as np
from math import exp
import random
import csv
import pickle
import numpy
import operator
import multiprocessing
import json
from operator import itemgetter

def createRandomDistribution(KVal):
    distribution = np.random.multivariate_normal(mean, cov, size = 1)
    return distribution[0]

# Initialize program parameters
# userCount = 0
lossSamples = []
users = {}
products = []
freqProducts = {}
userProducts = {}
usersLF = {}
K = 10
epsilon = 0.01
learning_rate=0.05
bias_regularization=1.0
user_regularization=0.0025
positive_item_regularization=0.0025
negative_item_regularization=0.00025
update_negative_item_factors=True
mean = np.zeros(K)
sigma = 0.1
lambdaVal = 0.01 #/ (sigma ** 2)
cov = np.diag(np.ones(K) * lambdaVal)

# Load all the products bought by each user
with open('data/transactionLog.csv') as transactionLog:
    #with open('/home/rcf-proj2/akr/akshitar/target/data/transactionLog.csv') as transactionLog:
    count = -1
    reader = csv.reader(transactionLog)
    for userData in reader:
        count = count + 1
        if (count == 0):
            continue
        if count % 10000000 == 0:
            print(count)
            if count >= 300000000:
                print(count)
                break
        uID = str(userData[0])
        skuID = str(userData[1])
        if uID in userProducts:
            userProducts[uID].append(skuID)
        else:
            userProducts[uID] = [skuID]

        if skuID in freqProducts:
            freqProducts[skuID] = freqProducts[skuID] + 1
        else:
            freqProducts[skuID] = 1

freqProdCount = sorted(freqProducts.items(), key = itemgetter(1), reverse = True)

index = 0
threshold = 1000
for x in range(0, len(freqProdCount)):
    if (threshold > freqProdCount[x][1]):
        index = x
        break

def convertListToDict(listObj):
    dictObj = dict()
    for x in range(0, len(listObj)):
        dictObj[listObj[x][0]] = listObj[x][1]
    return dictObj

freqProducts = convertListToDict(freqProdCount[:index])

products = freqProducts.keys()
item_bias = dict()
for x in products:
    item_bias[x] = 0;
print(len(products))

users = list(userProducts.keys())

print('Stored the loaded variables')

print('Create Random Distribution')
for x in users:
    usersLF[x] = createRandomDistribution(K)

productsLF = {}
for x in products:
    productsLF[x] = createRandomDistribution(K)

userCount = len(users)
lossSamplesNumber = int(100 * len(users)**0.5)

def getItemScore(user, product):
    score = 0

    # Add long term interest
    longTerm = np.dot(usersLF[user], productsLF[product])

    # Add short term interest
    # shortTerm = getShortTermScore(user, product)

    score = longTerm  #+ shortTerm
    return score

def vectorSubtract(vectorA, vectorB):
    return map(operator.sub, vectorA, vectorB)

def vectorAdd(vectorA, vectorB):
    return map(operator.add, vectorA, vectorB)

def vectorMultiply(vectorA, val):
    def timesTwo(x):
        return x * val
    return map(timesTwo, vectorA)

def createLossSamples():
    lossSamplesNumber = int(50 * len(users)**0.5)
    print('sampling {0} <user,item i,item j> triples...'.format(lossSamplesNumber))
    lossSamples = [t for t in generateSamples(lossSamplesNumber)]
    return lossSamples

def getRandomUser():
    return users[random.randint(0, len(users) - 1)]

def getUserPurchasedProduct(user):
    count = 0
    purchasedProducts = userProducts[user]
    while (True):
        count = count + 1
        i = purchasedProducts[random.randint(0, len(purchasedProducts) - 1)]
        if (i in products or count > 20):
            if (count > 20):
                i = -1
            break
    return i

def getNotPurchasedProduct(user):
    purchasedProducts = userProducts[user]
    selectedProduct = products[random.randint(0, len(products) - 1)]
    while selectedProduct in purchasedProducts:
        selectedProduct = products[random.randint(0, len(products) - 1)]
    return selectedProduct

def generateSamples(maxSampleSize):
    for _ in range(0,maxSampleSize):
        u = getRandomUser()
        i = getUserPurchasedProduct(u)
        while (i < 0) :
            u = getRandomUser()
            i = getUserPurchasedProduct(u)
        j = getNotPurchasedProduct(u)
        yield u,i,j

def loss(lossSamples):
    ranking_loss = 0
    for u,i,j in lossSamples:
        x = predict(u,i) - predict(u,j)
        ranking_loss += 1.0 / (1.0 + exp(x))

    complexity = 0
    for u,i,j in lossSamples:
        complexity += user_regularization * np.dot(usersLF[u], usersLF[u])
        complexity += positive_item_regularization * np.dot(productsLF[i], productsLF[i])
        complexity += negative_item_regularization * np.dot(productsLF[j], productsLF[j])
        complexity += bias_regularization * (item_bias[str(i)]**2)
        complexity += bias_regularization * (item_bias[str(j)]**2)

    return ranking_loss + (0.5 * complexity)

def update_factors(user, i, j):

    x = (item_bias[str(i)]-item_bias[str(j)])+ np.dot(usersLF[user], vectorSubtract(productsLF[i], productsLF[j])) # Add item bias
    z = 0.5
    try:
        z = 1.0 / (1.0 + exp(x))
    except:
        print('Z value fault')

    # update bias terms
    ditemi = z - bias_regularization * item_bias[i]
    item_bias[i] += (learning_rate * ditemi)

    ditemj = -z - bias_regularization*item_bias[j]
    item_bias[j] += (learning_rate * ditemj)

    # update_u:
    dU = vectorSubtract(vectorMultiply(vectorSubtract(productsLF[i], productsLF[j]), z), vectorMultiply(usersLF[user], user_regularization))
    usersLF[user] = vectorAdd(usersLF[user], vectorMultiply(dU, learning_rate))

    dI = vectorSubtract(vectorMultiply(usersLF[user],z), vectorMultiply(productsLF[i], positive_item_regularization))
    productsLF[i] = vectorAdd(productsLF[i], vectorMultiply(dI, learning_rate))

    dJ = vectorSubtract(vectorMultiply(usersLF[user], -z), vectorMultiply(productsLF[j], negative_item_regularization))
    productsLF[j] = vectorAdd(productsLF[j], vectorMultiply(dJ, learning_rate))


def predict(user,i):
    dummy = np.dot(usersLF[user], productsLF[i])
    return (dummy + item_bias[str(i)])# Add bias

def recObj(user, recommends):
    obj = dict()
    obj['guest_id'] = int(user)
    obj['recs'] = []
    for x in recommends:
        obj['recs'].append(int(x))
    return json.dumps(obj)

lossSamples = createLossSamples()
prevLoss = loss(lossSamples)
print("initial loss = {0}".format(prevLoss))
for it in xrange(500):
    print('starting iteration {0}'.format(it))
    for u,i,j in generateSamples(lossSamplesNumber):
        update_factors(u,i,j)
    newLoss = loss(lossSamples)
    print('iteration {0}: loss = {1}'.format(it, newLoss))
    if prevLoss < newLoss:
        break
    else:
        prevLoss = newLoss

print('Storing the latent factors')

workerCount = 40
totalUsers = len(users)

def writeToJSON(index):
    resultFile = open("jayResults.json", "a")
    for user in users[index * (totalUsers/workerCount): (index + 1) * (totalUsers/workerCount)]:
        results = []
        for prod in products:
            if not prod in userProducts[user]:
                results.append((prod, np.dot(productsLF[prod], usersLF[user])))
        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:5]
        finalResults = []
        for x in results:
            finalResults.append(int(x[0]))
        resultFile.write('\n' + recObj(user, finalResults))
    resultFile.close()


jobs = []
for i in range(workerCount):
    p = multiprocessing.Process(target=writeToJSON, args=(i,))
    jobs.append(p)
    p.start()
