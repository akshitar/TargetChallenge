from itertools import combinations

def ord_comb(l,n):
    return list(combinations(l,n))

minsup = 0.3
minconf = 0.8

def count_first(transactions):
    adict = {}
    for t in transactions:
        for item in t:
            if item in adict:
                adict[item] += 1
            else:
                adict[item] = 1
    return adict
def find_frequent(Candidate, minsup, no_of_lines):
    adict={}
    for key in Candidate:
        if ((float)(Candidate[key]/no_of_lines)) >= minsup:
            adict[key] = Candidate[key]
    return adict
def candidate_gen(keys):
    adict={}
    for i in keys:
        for j in keys:
            if i != j and (j,i) not in adict:
                adict[tuple([min(i,j),max(i,j)])] = 0
    return adict
def add_frequency(Candidate, transactions):
    for key in Candidate:
        for t in transactions:
            if key[0] in t and key[1] in t:
                Candidate[key] += 1
    return Candidate

transactions = [['A', 'B', 'C', 'D'], ['A', 'B', 'C', 'D']]
no_of_lines=2
print(no_of_lines)
#First iteration
C1 = count_first(transactions)
F1 = find_frequent(C1,minsup,no_of_lines)
#Second iteration
C2 = candidate_gen(F1.keys())
C2 = add_frequency(C2,transactions)
F2 = find_frequent(C2,minsup,no_of_lines)
print(F2)
