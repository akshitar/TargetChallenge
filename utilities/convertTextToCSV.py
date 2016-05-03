import re
import csv
with open('data/skuFeatures.csv', 'w') as fp:
    a = csv.writer(fp, delimiter=',')
    a.writerow(['SKU', 'Features'])
    with open('data/usc_item_anon_attr') as f:
        for line in f:
            s = line.split('	')
            a.writerow([s[0], s[1]])
