import operator
import numpy as np
import math
import random
from operator import itemgetter

def sigmaFunction(val):
    value = 1 / (1 + math.exp(-val))
    return value

# Populate with all products
# products = []
K = 50
mean = np.zeros(K)
sigma = 0.1
lambdaVal = 0.01 #/ (sigma ** 2)
cov = np.diag(np.ones(K) * lambdaVal)

def createRandomDistribution(KVal):
    distribution = np.random.multivariate_normal(mean, cov, size = 1)
    return distribution[0]

users = dict()
users['Jay'] = ['A', 'B']
epsilon = 0.01
latentFactorUsers = {
    'Jay': createRandomDistribution(K)
}
latentFactorProducts = {
    'A': createRandomDistribution(K),
    'B': createRandomDistribution(K),
    'C': createRandomDistribution(K),
    'P': createRandomDistribution(K),
    'Q': createRandomDistribution(K),
    'X': createRandomDistribution(K),
    'Y': createRandomDistribution(K),
    'Z': createRandomDistribution(K),
    'AA': createRandomDistribution(K),
    'AB': createRandomDistribution(K),
    'AC': createRandomDistribution(K),
    'AD': createRandomDistribution(K),
    'AE': createRandomDistribution(K),
    'AF': createRandomDistribution(K),
    'AG': createRandomDistribution(K)
}
products = ['A', 'B', 'C', 'P', 'Q', 'X', 'Y', 'Z', 'AA', 'AB', 'AC', 'AD', 'AE', 'AF', 'AG']

def generateRandom(max):
    return random.randint(0, max)

def vectorSubtract(vectorA, vectorB):
    return map(operator.sub, vectorA, vectorB)

def vectorAdd(vectorA, vectorB):
    return map(operator.add, vectorA, vectorB)

def vectorMultiply(vectorA, val):
    def timesTwo(x):
        return x * val
    return map(timesTwo, vectorA)

def BPR(users):
    bprValue = 1
    for user in users:
        userProducts = users[user]
        for purchase in userProducts:
            for prod in products:
                if not prod in userProducts:
                    bprValue = bprValue * sigmaFunction(dot(latentFactorUsers[user], latentFactorProducts[purchase]) - dot(latentFactorUsers[user], latentFactorProducts[prod]))

# def higherOrderAffinity():
def SGD(users):
    for user in users:
        userProducts = users[user]
        for x in range(0, 5):
            for purchase in userProducts:
                for prod in products:
                    if not prod in userProducts:
                        cUIJ = 1 - sigmaFunction(np.dot(latentFactorUsers[user], latentFactorProducts[purchase]) - np.dot(latentFactorUsers[user], latentFactorProducts[prod]))

                        diffU = vectorAdd(vectorMultiply(vectorSubtract(latentFactorProducts[purchase], latentFactorProducts[prod]), cUIJ), vectorMultiply(latentFactorUsers[user], lambdaVal))
                        diffVI = vectorAdd(vectorMultiply(latentFactorUsers[user], cUIJ), vectorMultiply(latentFactorProducts[purchase], lambdaVal))
                        diffVJ = vectorAdd(vectorMultiply(latentFactorUsers[user], -1 * cUIJ), vectorMultiply(latentFactorProducts[prod], lambdaVal))

                        latentFactorUsers[user] = vectorAdd(latentFactorUsers[user], vectorMultiply(diffU, epsilon))
                        latentFactorProducts[purchase] = vectorAdd(latentFactorProducts[purchase], vectorMultiply(diffVI, epsilon))
                        latentFactorProducts[prod] = vectorAdd(latentFactorProducts[prod], vectorMultiply(diffVJ, epsilon))

SGD(users)
values = {}
for x in latentFactorProducts:
    values[x] = np.dot(latentFactorProducts[x], latentFactorUsers['Jay'])

print(sorted(values.items(), key = itemgetter(1), reverse = True))
