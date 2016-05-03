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
products = {}
userProducts = {}
usersLF = {}

# Load all the products bought by each user
with open('data/transactionLog.csv') as transactionLog:
    #with open('/home/rcf-proj2/akr/akshitar/target/data/transactionLog.csv') as transactionLog:
    count = -1
    reader = csv.reader(transactionLog)
    for userData in reader:
        count = count + 1
        if (count == 0):
            continue
        uID = str(userData[0])
        skuID = str(userData[1])
        if skuID in products:
            products[skuID] = products[skuID] + 1
        else:
            products[skuID] = 1
productsSorted = sorted(products.items(), key=itemgetter(1), reverse = True)
print(productsSorted)
