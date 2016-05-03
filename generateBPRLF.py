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
import psycopg2
count = 0
try:
    conn = psycopg2.connect("dbname='target' user='target' host='targetdatascience.cxbkly7up5ho.us-east-1.rds.amazonaws.com' password='password' port='5432'")
except Exception as e:
    print(e)
    print "I am unable to connect to the database"

# Initialize BPR Parameters
K = 50
learning_rate = 0.05
bias_regularization = 1.0
user_regularization = 0.0025
positive_item_regularization = 0.0025
negative_item_regularization = 0.00025
mean = np.zeros(K)
sigma = 0.1
lambdaVal = 0.01 #/ (sigma ** 2)
cov = np.diag(np.ones(K) * lambdaVal)

def createRandomDistribution(KVal):
    distribution = np.random.multivariate_normal(mean, cov, size = 1)
    return distribution[0]

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

fileProductFeatures = 'data/skuFeatures'
fileTransactionLog = 'data/transactionLog.csv'

# Load all the products bought by each user
with open(fileTransactionLog, 'rb') as transactionLog:
    count = -1
    reader = csv.reader(transactionLog)
    for userData in reader:
        count = count + 1
        if (count == 0):
            continue
        if count % 10000000 == 0:
            print(count)
        uID = str(userData[0])
        skuID = str(userData[1])
        if uID in userProducts:
            userProducts[uID].append(skuID)
        else:
            userProducts[uID] = [skuID]
        products.append(skuID)


try:
    pickle.dump(products, fileProducts)
    # products = pickle.load(fileProducts)

    # pickle.dump(userProducts, fileUserProducts)
    print('Store the loaded variables')
    # userProducts = pickle.load(fileUserProducts)
except:
    print('Store failed')

products = list(set(products))
users = userProducts.keys()
print('Done')
print('Create Random Distribution')
for x in users:
    usersLF[x] = createRandomDistribution(K)

productsLF = {}
for x in products:
    productsLF[x] = createRandomDistribution(K)

userCount = len(users)
lossSamplesNumber = int(100 * len(users)**0.5)

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

def vectorSubtract(vectorA, vectorB):
    return map(operator.sub, vectorA, vectorB)

def vectorAdd(vectorA, vectorB):
    return map(operator.add, vectorA, vectorB)

def vectorMultiply(vectorA, val):
    def timesTwo(x):
        return x * val
    return map(timesTwo, vectorA)

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

def update_factors(user, i, j):
    x = np.dot(usersLF[user], vectorSubtract(productsLF[i], productsLF[j])) # Add item bias
    z = 1
    try:
        z = 1.0 / (1.0 + exp(x))
    except:
        print('Z value fault')

    # update_u:
    d = vectorSubtract(vectorMultiply(vectorSubtract(productsLF[i], productsLF[j]), z), vectorMultiply(usersLF[user], user_regularization))
    usersLF[user] = vectorAdd(usersLF[user], vectorMultiply(d, learning_rate))

    # update_i:
    d = vectorSubtract(vectorMultiply(usersLF[user], z), vectorMultiply(productsLF[i], positive_item_regularization))
    productsLF[i] = vectorAdd(productsLF[i], vectorMultiply(d, learning_rate))

    # update_j:
    d = vectorSubtract(vectorMultiply(usersLF[user], -z), vectorMultiply(productsLF[j], negative_item_regularization))
    productsLF[j] = vectorAdd(productsLF[j], vectorMultiply(d, learning_rate))

def predict(user,i):
    return np.dot(usersLF[user], productsLF[i]) # Add bias

def recObj(user, recommends):
    obj = dict()
    obj['guest_id'] = int(user)
    obj['recs'] = []
    for x in recommends:
        obj['recs'].append(int(x))
    return json.dumps(obj)

lossSamples = createLossSamples()
prevLoss = loss(lossSamples)
print 'initial loss = {0}'.format(prevLoss)
for it in xrange(5):
    print 'starting iteration {0}'.format(it)
    for u,i,j in generateSamples(lossSamplesNumber):
        update_factors(u,i,j)
    newLoss = loss(lossSamples)
    print 'iteration {0}: loss = {1}'.format(it, newLoss)
    if prevLoss < newLoss:
        break
    else:
        prevLoss = newLoss

def clearLF():
    cur = conn.cursor()
    statement = "DELETE FROM \"usersLFN\""
    cur.execute(statement)

    statement = "DELETE FROM \"productsLFN\""
    cur.execute(statement)

    conn.commit()

def saveUserLF():
    count = 0
    cur = conn.cursor()
    for user in usersLF:
        values = usersLF[user]
        values = ",".join(str(i) for i in values)
        statement = "INSERT INTO \"usersLFN\"(\"userName\", \"LF\") VALUES ('%s', '%s')" % (user, values)
        cur.execute(statement)
        count = count + 1
        if count % 10000 == 0:
            conn.commit()
            print(count)
    conn.commit()

def saveProductLF():
    count = 0
    cur = conn.cursor()
    for prod in productsLF:
        values = productsLF[prod]
        values = ",".join(str(i) for i in values)
        statement = "INSERT INTO \"productsLFN\"(\"productName\", \"LF\") VALUES ('%s', '%s')" % (prod, values)
        cur.execute(statement)
        count = count + 1
        if count % 10000 == 0:
            conn.commit()
            print(count)
    conn.commit()


try:
    print('Clearing the previous latent factors')
    clearLF()

    print('Storing the latent factors')
    print('Saving product LF')
    saveProductLF()

    print('Saving user LF')
    saveUserLF()

except Exception as e:
    print(e)
    print('Store failed for latent factors')
