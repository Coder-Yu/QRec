from numpy.linalg import norm
from math import sqrt,exp
from numba import jit
import heapq

def l1(x):
    return norm(x,ord=1)

def l2(x):
    return norm(x)

def common(x1,x2):
    # find common ratings
    common = (x1!=0)&(x2!=0)
    new_x1 = x1[common]
    new_x2 = x2[common]
    return new_x1,new_x2

def cosine_sp(x1,x2):
    'x1,x2 are dicts,this version is for sparse representation'
    total = 0
    denom1 = 0
    denom2 =0
    try:
        for k in x1:
            if k in x2:
                total+=x1[k]*x2[k]
                denom1+=x1[k]**2
                denom2+=x2[k]**2
        return total/(sqrt(denom1) * sqrt(denom2))
    except ZeroDivisionError:
        return 0

def euclidean_sp(x1,x2):
    'x1,x2 are dicts,this version is for sparse representation'
    total = 0
    try:
        for k in x1:
            if k in x2:
                total+=x1[k]**2-x2[k]**2
        return 1/total
    except ZeroDivisionError:
        return 0

def cosine(x1,x2):
    #find common ratings
    #new_x1, new_x2 = common(x1,x2)
    #compute the cosine similarity between two vectors
    sum = x1.dot(x2)
    denom = sqrt(x1.dot(x1)*x2.dot(x2))
    try:
        return sum/denom
    except ZeroDivisionError:
        return 0

    #return cosine_similarity(x1,x2)[0][0]

def pearson_sp(x1,x2):
    total = 0
    denom1 = 0
    denom2 = 0
    overlapped=False
    try:
        mean1 = sum(x1.values())/len(x1)
        mean2 = sum(x2.values()) /len(x2)
        for k in x1:
            if k in x2:
                total += (x1[k]-mean1) * (x2[k]-mean2)
                denom1 += (x1[k]-mean1) ** 2
                denom2 += (x2[k]-mean2) ** 2
                overlapped=True
        return total/ (sqrt(denom1) * sqrt(denom2))
    except ZeroDivisionError:
        if overlapped:
            return 1
        return 0

def euclidean(x1,x2):
    #find common ratings
    new_x1, new_x2 = common(x1, x2)
    #compute the euclidean between two vectors
    diff = new_x1-new_x2
    denom = sqrt((diff.dot(diff)))
    try:
        return 1/denom
    except ZeroDivisionError:
        return 0


def pearson(x1,x2):
    #find common ratings
    #new_x1, new_x2 = common(x1, x2)
    #compute the pearson similarity between two vectors
    #ind1 = new_x1 > 0
    #ind2 = new_x2 > 0
    try:
        mean_x1 = x1.sum()/len(x1)
        mean_x2 = x2.sum()/len(x2)
        new_x1 = x1 - mean_x1
        new_x2 = x2 - mean_x2
        sum = new_x1.dot(new_x2)
        denom = sqrt((new_x1.dot(new_x1))*(new_x2.dot(new_x2)))
        return sum/denom
    except ZeroDivisionError:
        return 0


def similarity(x1,x2,sim):
    if sim == 'pcc':
        return pearson_sp(x1,x2)
    if sim == 'euclidean':
        return euclidean_sp(x1,x2)
    else:
        return cosine_sp(x1, x2)


def normalize(vec,maxVal,minVal):
    'get the normalized value using min-max normalization'
    if maxVal > minVal:
        return (vec-minVal)/(maxVal-minVal)
    elif maxVal==minVal:
        return vec/maxVal
    else:
        print('error... maximum value is less than minimum value.')
        raise ArithmeticError

def sigmoid(val):
    return 1/(1+exp(-val))


def denormalize(vec,maxVal,minVal):
    return minVal+(vec-0.01)*(maxVal-minVal)

@jit(nopython=True)
def find_k_largest(K, candidates):
    n_candidates = []
    for iid, score in enumerate(candidates[:K]):
        n_candidates.append((score, iid))
    heapq.heapify(n_candidates)
    for iid, score in enumerate(candidates[K:]):
        if score > n_candidates[0][0]:
            heapq.heapreplace(n_candidates, (score, iid + K))
    n_candidates.sort(key=lambda d: d[0], reverse=True)
    ids = [item[1] for item in n_candidates]
    k_largest_scores = [item[0] for item in n_candidates]
    return ids, k_largest_scores
