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
import psycopg2
count = 0
try:
    conn = psycopg2.connect("dbname='target' user='target' host='targetdatascience.cxbkly7up5ho.us-east-1.rds.amazonaws.com' password='password' port='5432'")
except Exception as e:
    print(e)
    print "I am unable to connect to the database"

fileProductFeatures = 'data/skuFeatures'
fileTransactionLog = 'data/transactionLog.csv'
userProducts = dict()
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

def storeUserProducts():
    count = 0
    cur = conn.cursor()
    for user in userProducts:
        products = userProducts[user]
        statement = "INSERT INTO \"userProducts\"(\"user\", \"products\") VALUES ('%s', '%s')" % (user, ','.join(products))
        cur.execute(statement)
        count = count + 1
        if count % 1000000 == 0:
            conn.commit()
            print(count)
    conn.commit()

storeUserProducts()
