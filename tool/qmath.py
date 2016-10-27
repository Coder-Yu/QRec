from sklearn.metrics.pairwise import pairwise_distances,cosine_similarity
import numpy as np
from numpy.linalg import norm
from scipy.stats.stats import pearsonr
from math import sqrt

def cosine(x1,x2):
    #find common ratings
    common = (x1*x2)>0
    new_x1 = x1[common]
    new_x2 = x2[common]
    #compute the similarity between two vectors
    # sum = 0
    # denom = 0
    # for i in range(len(new_x1)):
    sum = new_x1.dot(new_x2.transpose())
    denom = sqrt((new_x1*new_x1).sum()+(new_x2*new_x2).sum())
    try:
        return float(sum)/denom
    except ZeroDivisionError:
        return 0

    #return cosine_similarity(x1,x2)[0][0]

def l1(x):
    return norm(x,ord=1)

def l2(x):
    return norm(x)

def euclidean(x1,x2):
    return pairwise_distances(x1,x2,metric='euclidean')[0][0]


def pearson(x1,x2):
    #find common ratings
    common = (x1*x2)>0
    new_x1 = x1[common]
    new_x2 = x2[common]
    #compute the similarity between two vectors
    return pearsonr(new_x1,new_x2)[0]


def similarity(x1,x2,sim):
    if sim == 'pcc':
        return pearson(x1,x2)
    if sim == 'euclidean':
        return euclidean(x1,x2)
    else:
        return cosine(x1, x2)


def normalize(vec,maxVal,minVal):
    'get the normalized value using min-max normalization'
    if maxVal > minVal:
        return (vec-minVal)/(maxVal-minVal)
    elif maxVal==minVal:
        return vec/maxVal
    else:
        print 'error... maximum value is less than minimum value.'
        raise ArithmeticError

def denormalize(vec,maxVal,minVal):
    return minVal+vec*(maxVal-minVal)
