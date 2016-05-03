import csv
import json
import pickle
from operator import itemgetter

fileExistingFeatures = open('assets/existingFeatures', 'r+')
fileFeatureProducts = open('assets/featureProducts', 'r+')
fileFeatureCount = open('assets/featureCount', 'r+')
fileProductFeatures = open('assets/productFeatures', 'r+')
fileTaxonomyTree = open('assets/taxonomyTree', 'r+')

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
            current_dict = current_dict.setdefault(letter, {})
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

class Node(object):
    def __init__(self, name, children, parent):
        self.name = name
        self.children = []
        self.count = 0
        self.parent = parent
        if children is not None:
            for child in children:
                self.addChild(child)

    def __repr__(self):
        return self.name + ' - Count:' + str(self.count)

    def addChild(self, childNode):
        assert isinstance(childNode, Node)
        self.children.append(childNode)

    def addParent(self, parentNode):
        assert isinstance(childNode, Node)
        self.parent = childNode

    def incrementCount(self):
        self.count = self.count + 1

    def printChildren(self):
        for child in self.children:
            print(child)

existingFeatures = dict()
featureProducts = dict()
productFeatures = dict()
#
# root = Node('root', None, None)
# existingFeatures['root'] = root

# Get count for all features
# Associate each product to the corresponding features
with open('data/skuFeatures.csv', 'rb') as features:
    count = -1
    reader = csv.reader(features)
    for row in reader:
        count = count + 1
        if count % 1000 == 0:
            print(count)
        if count != 0:
            featureList = cleanArr(row[1].strip().split(' '))
            skuID = row[0]

            # Store all the features in existingFeatures{}
            for feature in set(featureList):
                if feature in existingFeatures:
                    existingFeatures[feature] = existingFeatures[feature] + 1
                else:
                    existingFeatures[feature] = 1

                # if feature in featureProducts:
                #     featureProducts[feature] = featureProducts[feature] + [skuID]
                # else:
                #     featureProducts[feature] = [skuID]

# Convert the feature dict to a list ranked in descending order of feature count
# pickle.dump(existingFeatures, fileFeatureCount)
# sortFeatures = sorted(existingFeatures.items(), key=itemgetter(1), reverse = True)
# pickle.dump(sortFeatures, fileExistingFeatures)

# Save the featureProduct dictionary
# pickle.dump(featureProducts, fileFeatureProducts)

featureCount = pickle.load(fileFeatureCount)
featureCount = sorted(featureCount.items(), key = itemgetter(1), reverse = True)

index = 0
threshold = 75000
for x in range(0, len(featureCount)):
    if (threshold > featureCount[x][1]):
        index = x
        break

features = featureCount[index:]
stopwords = featureCount[:index]
stopwords = convertListToDict(stopwords)
features = convertListToDict(features)
featureCount = convertListToDict(featureCount)
# existingFeatures = pickle.load(fileExistingFeatures)
# print(existingFeatures)
#
# featureProducts = pickle.load(fileFeatureProducts)
# print(featureProducts)

# Sort the features for each product based on the count
trieArr = []
with open('data/skuFeatures.csv', 'rb') as features:
    count = -1
    reader = csv.reader(features)
    for row in reader:
        count = count + 1
        if count > 50:
            if count % 10 == 0:
                print(count)
                if count > 100:
                    break
            if count != 0:
                tempArr = list(set(cleanArr(row[1].split(' '))))
                tempArr.sort(rankFeaturesByCount)
                tempArr = removeStopWords(tempArr)
                productFeatures[row[0]] = tempArr
                trieArr.append(tempArr)

# pickle.dump(productFeatures, fileProductFeatures)
# productFeatures = pickle.load(fileProductFeatures)

# Iterate through all the features and build the tree
taxonomyTree = make_trie(trieArr)
print(json.dumps(taxonomyTree))
# pickle.dump(taxonomyTree, fileTaxonomyTree)
