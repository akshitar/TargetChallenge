import csv
import json
import pickle
from operator import itemgetter

fileExistingFeatures = open('assets/existingFeatures', 'r+')
fileFeatureProducts = open('assets/featureProducts', 'r+')
fileFeatureCount = open('assets/featureCount', 'r+')
fileProductFeatures = open('assets/productFeatures', 'r+')
fileTaxonomyTree = open('assets/taxonomyTree', 'r+')

featureCount = pickle.load(fileFeatureCount)
sortFeatures = sorted(featureCount.items(), key=itemgetter(1), reverse = True)
print(sortFeatures)
