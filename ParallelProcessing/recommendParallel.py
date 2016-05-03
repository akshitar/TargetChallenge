from math import exp
import numpy as np
import random
import cPickle as pickle
import numpy
import operator
import csv
from operator import itemgetter
import pickle
import json
import sys
import psycopg2
import multiprocessing

try:
    conn = psycopg2.connect("dbname='target' user='target' host='targetdatascience.cxbkly7up5ho.us-east-1.rds.amazonaws.com' password='password' port='5432'")
except Exception as e:
    print(e)
    print "I am unable to connect to the database"

count = 0

# Initialize program parameters
userCount = 0
lossSamples = []
users = []
products = []
userProducts = {}
usersLF = {}
productsLF = {}

fileProductFeatures = 'data/skuFeatures'
fileTransactionLog = 'data/transactionLog.csv'

def recObj(user, recommends):
    obj = dict()
    obj['guest_id'] = int(user)
    obj['recs'] = []
    for x in recommends:
        obj['recs'].append(int(x))
    return json.dumps(obj)

# totalUsers = 20 * 1000000
totalUsers = 10000000
bins = 100
count = int(sys.argv[1])
print(count)

def loadAllProducts():
    cur = conn.cursor()
    cur.execute("SELECT * FROM \"productsLFN\"")
    rows = cur.fetchall()
    for prod in rows:
        pID = str(prod[0])
        productsLF[pID] = []
        values = prod[1].split(',')
        for x in values:
            productsLF[pID].append(float(x))
    print('Products LF loaded')


def loadUserProducts():
    cur = conn.cursor()
    cur.execute("SELECT * FROM \"userProducts\" LIMIT %d OFFSET %d" % ((totalUsers/bins), (count) * (totalUsers/bins)))
    rows = cur.fetchall()
    for user in rows:
        userProducts[user[0]] = user[1].split(',')
    print('User products loaded')

def loadAllUsers():
    loadedUsers = userProducts.keys()
    composeString = '\''
    for x in loadedUsers:
        composeString = composeString + (x + '\',\'')
    composeString = composeString[:len(composeString) - 2]
    cur = conn.cursor()
    statement = "SELECT * FROM \"usersLFN\" WHERE \"userName\" IN (%s)" % (composeString)
    cur.execute(statement)
    rows = cur.fetchall()
    for user in rows:
        uID = user[0]
        usersLF[uID] = []
        values = user[1].split(',')
        for x in values:
            usersLF[uID].append(float(x))
    print('User LF loaded')

print('Loading the latent factors')

loadAllProducts()
loadUserProducts()
loadAllUsers()

products = productsLF.keys()
print('Converting to the keys')
products = productsLF.keys()
allUsers = usersLF.keys()
users = usersLF.keys()

print(len(products))
workerCount = 40
totalUsers = len(users)


def writeToJSON(index):
    resultFile = open("recommend/targetRecommendationsN1-" + str(count) + ".json", "a")
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
        resultFile.write(recObj(user, finalResults) + '\n')
    resultFile.close()


jobs = []
for i in range(workerCount):
    p = multiprocessing.Process(target=writeToJSON, args=(i,))
    jobs.append(p)
    p.start()
