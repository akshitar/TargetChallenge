from math import exp
import numpy as np
import random
import cPickle as pickle
import numpy
import operator
import csv
from operator import itemgetter
import marshal as pickle
import json
import math

K = 10
alpha = 0.1

def getAlphaValue(sN, bN):
    return alpha * (math.exp(-sN / bN))

def createRandomDistribution(KVal):
    distribution = np.random.multivariate_normal(mean, cov, size = 1)
    return distribution[0]

# Build taxonomyTree
def convertListToDict(listObj):
    dictObj = dict()
    for x in range(0, len(listObj)):
        dictObj[listObj[x][0]] = listObj[x][1]
    return dictObj

_end = '_end_'
def make_trie(words):
    print(len(words))
    root = dict()
    for word in words:
        current_dict = root
        for letter in word:
            current_dict = current_dict.setdefault(letter, {'weight': createRandomDistribution(K)})
        current_dict[_end] = _end
    return root

def rankFeaturesByCount(a, b):
    if featureCount[a] > featureCount[b]:
        return -1
    elif featureCount[b] > featureCount[a]:
        return 1
    else:
        return 0

def cleanArr(arr):
    for i in range(0, len(arr)):
        arr[i] = arr[i].replace('token_', 't')
        arr[i] = arr[i].strip()
    return arr

def removeStopWords(featureList):
    for x in range(0, len(featureList)):
        if not featureList[x] in stopwords:
            featureList = featureList[x:]
            return featureList
    return featureList

fileProductFeatures = open('assets/productFeatures', 'r+')
productFeatures = pickle.load(fileProductFeatures)
taxonomyTree = make_trie(trieArr)

resultFile = open("targetRecommendations-ShortTerm.json", "w")
# Initialize BPR Parameters
learning_rate = 0.05
bias_regularization = 1.0
user_regularization = 0.0025
positive_item_regularization = 0.0025
negative_item_regularization = 0.00025
mean = np.zeros(K)
sigma = 0.1
lambdaVal = 0.01 #/ (sigma ** 2)
cov = np.diag(np.ones(K) * lambdaVal)

def vectorSubtract(vectorA, vectorB):
    return map(operator.sub, vectorA, vectorB)

def vectorAdd(vectorA, vectorB):
    return map(operator.add, vectorA, vectorB)

def vectorMultiply(vectorA, val):
    def timesTwo(x):
        return x * val
    return map(timesTwo, vectorA)

def getProductVector(product):
    featureVector = np.zeros(K)
    features = productFeatures[product]
    treeObj = taxonomyTree
    for x in features:
        featureVector = vectorAdd(featureVector, treeObj[x]['weight'])
        treeObj = treeObj[x]
    return featureVector

# Initialize program parameters
userCount = 0
lossSamples = []
users = []
products = []
userProducts = {}
usersLF = {}
productsLF = {}


fileProducts = open('assets/productsR', 'r+')
fileUserProducts = open('assets/userProductsR', 'r+')
fileRecommendations = open('assets/recommendations', 'r+')

fileUsersLF = open('assets/usersLF', 'r+')
fileProductsLF = open('assets/productsLF', 'r+')
fileUserPurchaseChronology = open('assets/userPurchaseChronology', 'r+')

fileProductFeatures = 'data/skuFeatures'
fileTransactionLog = 'data/transactionLog.csv'

userChronology = pickle.load(fileUserPurchaseChronology)
for user in userChronology:
    userChronology[user] = reversed(userChronology[user])

# Load all the products bought by each user
with open(fileTransactionLog, 'rb') as transactionLog:
    count = -1
    reader = csv.reader(transactionLog)
    for userData in reader:
        count = count + 1
        if (count == 0):
            continue
        if count % 1000000 == 0:
            print(count)
            if count > 10000000:
                break
        uID = str(userData[0])
        skuID = str(userData[1])
        if uID in userProducts:
            userProducts[uID].append(skuID)
        else:
            userProducts[uID] = [skuID]
        products.append(skuID)

products = list(set(products))
users = userProducts.keys()

print('Store the loaded variables')
pickle.dump(products, fileProducts)
pickle.dump(userProducts, fileUserProducts)

print('Create Random Distribution')
for x in users:
    usersLF[x] = createRandomDistribution(K)

productsLF = {}
for x in products:
    productsLF[x] = createRandomDistribution(K)

userCount = len(users)
lossSamplesNumber = int(100 * len(users)**0.5)

def getShortTermScore(user, product):
    shortTerm = 0
    count = 0
    purchasedProducts = userChronology[user]
    for trip in purchasedProducts:
        for item in trip:
            shortTerm = shortTerm + (alpha * np.dot(getProductVector(product), getProductVector(item)))
            count = count + 1
    shortTerm = shortTerm / count
    return shortTerm


def getItemScore(user, product):
    score = 0

    # Add long term interest
    longTerm = np.dot(usersLF[user], getProductVector(product))

    # Add short term interest
    shortTerm = getShortTermScore(user, product)

    score = longTerm + shortTerm
    return score

def createLossSamples():
    lossSamplesNumber = int(100 * len(users)**0.5)
    print 'sampling {0} <user,item i,item j> triples...'.format(lossSamplesNumber)
    lossSamples = [t for t in generateSamples(lossSamplesNumber)]
    return lossSamples

def getRandomUser():
    return users[random.randint(0, len(users) - 1)]

def getUserPurchasedProduct(user):
    purchasedProducts = userProducts[user]
    return purchasedProducts[random.randint(0, len(purchasedProducts) - 1)]

def getNotPurchasedProduct(user):
    purchasedProducts = userProducts[user]
    selectedProduct = products[random.randint(0, len(products) - 1)]
    while selectedProduct in purchasedProducts:
        selectedProduct = products[random.randint(0, len(products) - 1)]
    return selectedProduct

def generateSamples(maxSampleSize):
    for _ in xrange(maxSampleSize):
        u = getRandomUser()
        i = getUserPurchasedProduct(u)
        j = getNotPurchasedProduct(u)
        yield u,i,j

def loss(lossSamples):
    ranking_loss = 0;
    for u,i,j in lossSamples:
        x = predict(u,i) - predict(u,j)
        ranking_loss += 1.0 / (1.0 + exp(x))

    complexity = 0;
    for u,i,j in lossSamples:
        complexity += user_regularization * np.dot(usersLF[u], usersLF[u])
        complexity += positive_item_regularization * np.dot(productsLF[i], productsLF[i])
        complexity += negative_item_regularization * np.dot(productsLF[j], productsLF[j])
        # complexity += self.bias_regularization * self.item_bias[i]**2
        # complexity += self.bias_regularization * self.item_bias[j]**2

    return ranking_loss + 0.5 * complexity

def updateNodeWeights(product, updateVector):
    features = productFeatures[product]
    treeObj = taxonomyTree
    for x in features:
        treeObj[x]['weight'] = vectorAdd(treeObj[x]['weight'], updateVector)
        treeObj = treeObj[x]

def getPreviousPurchasedVector(user):
    previousPurchaseVector = np.zeros(K)
    previousPurchases = userChronology[user]
    for trip in previousPurchases:
        tripVector = np.zeros(K)
        tripLength = len(trip)
        for item in trip:
            tripVector = vectorAdd(tripVector, getProductVector(item))
        previousPurchaseVector = vectorAdd(previousPurchaseVector, vectorMultiply(tripVector, (alpha / tripLength)))
    return previousPurchaseVector

def getAlphaScore(user):
    score = 0
    purchases = userChronology[user]
    bN = len(purchases)
    for x in range(0, len(purchases)):
        score = score + (getAlphaValue(x + 1, bN) / len(purchases[x]))
    return score

def update_factors(user, i, j):
    x = np.dot(usersLF[user], vectorSubtract(getProductVector(i), getProductVector(j))) # Add item bias
    z = 0.5
    try:
        z = 1.0 / (1.0 + exp(x))
    except:
        print('Z value fault')

    # update_u:
    dU = vectorSubtract(vectorMultiply(vectorSubtract(getProductVector(i), getProductVector(j)), z), vectorMultiply(usersLF[user], user_regularization))
    usersLF[user] = vectorAdd(usersLF[user], vectorMultiply(dU, learning_rate))

    # update_i:
    dI = vectorMultiply(vectorSubtract(vectorSubtract(), vectorMultiply(getPreviousPurchasedVector(user), positive_item_regularization)) , z)
    updateNodeWeights(i, vectorMultiply(dI, learning_rate))

    # update_j:
    dJ = vectorMultiply(dI, -1)
    updateNodeWeights(j, vectorMultiply(dJ, learning_rate))

    dL = vectorSubtract(vectorMultiply(vectorSubtract(getProductVector(i), getProductVector(j)), z * getAlphaScore(user)), vectorMultiply(usersLF[user], user_regularization))
    purchases = userChronology[user]
    for trip in purchases:
        for item in trip:
            updateNodeWeights(item, vectorMultiply(dL, learning_rate))

def predict(user,i):
    return np.dot(usersLF[user], productsLF[i]) # Add bias

def recObj(user, recommends):
    obj = dict()
    obj['guest_id'] = user
    obj['recs'] = []
    for x in recommends:
        obj['recs'].append(int(x))
    return json.dumps(obj)

lossSamples = createLossSamples()
prevLoss = loss(lossSamples)
print 'initial loss = {0}'.format(prevLoss)
for it in xrange(500):
    print 'starting iteration {0}'.format(it)
    for u,i,j in generateSamples(lossSamplesNumber):
        update_factors(u,i,j)
    newLoss = loss(lossSamples)
    print 'iteration {0}: loss = {1}'.format(it, newLoss)
    if prevLoss < newLoss:
        break
    else:
        prevLoss = newLoss

print('Storing the latent factors')
pickle.dump(usersLF, fileUsersLF)
pickle.dump(productsLF, fileProductsLF)

recommendations = {}
# Generate the results for every user
for user in users:
    print('For user: ' + user)
    results = {}
    for prod in products:
        if not prod in userProducts[user]:
            results[prod] = np.dot(productsLF[prod], usersLF[user])
    results = sorted(results.items(), key = itemgetter(1), reverse = True)
    results = results[:5]
    recommendations[user] = []
    for x in results:
        recommendations[user].append(x[0])
    print(recommendations[user])
    resultFile.write(recObj(user, recommendations[user]) + '\n')
    print('--------------------')
resultFile.close()
