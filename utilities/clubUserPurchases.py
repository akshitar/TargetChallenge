import csv
import marshal as pickle


fileUserPurchase = open('assets/userPurchaseChronology', 'r+')
userPurchases = {}
prevUID = ''
with open('data/transactionLog.csv', 'rb') as transactionLog:
    count = -1
    reader = csv.reader(transactionLog)
    purchaseLog = []
    for userData in reader:
        count = count + 1
        if (count == 0):
            continue
        if count % 10000000 == 0:
            print(count)
        uID = str(userData[0])
        skuID = str(userData[1])


        if prevUID != '' and uID != prevUID:
            if prevUID in userPurchases:
                userPurchases[prevUID].append(purchaseLog)
            else:
                userPurchases[prevUID] = [purchaseLog]
            prevUID = uID
            purchaseLog = [skuID]
        else:
            prevUID = uID
            purchaseLog.append(skuID)
pickle.dump(userPurchases, fileUserPurchase)
