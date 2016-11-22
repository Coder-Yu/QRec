from sklearn.metrics.pairwise import pairwise_distances,cosine_similarity
import numpy as np
from numpy.linalg import norm
from scipy.stats.stats import pearsonr
from math import sqrt,exp

def l1(x):
    return norm(x,ord=1)

def l2(x):
    return norm(x)

def common(x1,x2):
    # find common ratings
    common = (x1<>0)&(x2<>0)
    new_x1 = x1[common]
    new_x2 = x2[common]
    return new_x1,new_x2

def cosine(x1,x2):
    #find common ratings
    new_x1, new_x2 = common(x1,x2)
    #compute the cosine similarity between two vectors
    sum = new_x1.dot(new_x2.transpose())
    denom = sqrt((new_x1.dot(new_x1.transpose()))+(new_x2.dot(new_x2.transpose())))
    try:
        return float(sum)/denom
    except ZeroDivisionError:
        return 0

    #return cosine_similarity(x1,x2)[0][0]


def euclidean(x1,x2):
    #find common ratings
    new_x1, new_x2 = common(x1, x2)
    #compute the euclidean between two vectors
    diff = new_x1-new_x2
    denom = sqrt((diff.dot(diff.transpose())))
    try:
        return 1/denom
    except ZeroDivisionError:
        return 0


def pearson(x1,x2):
    #find common ratings
    new_x1, new_x2 = common(x1, x2)
    #compute the pearson similarity between two vectors
    ind1 = new_x1 > 0
    ind2 = new_x2 > 0
    try:
        mean_x1 = float(new_x1.sum())/ind1.sum()
        mean_x2 = float(new_x2.sum())/ind2.sum()
        new_x1 = new_x1 - mean_x1
        new_x2 = new_x2 - mean_x2
        sum = new_x1.dot(new_x2.transpose())
        denom = sqrt((new_x1.dot(new_x1.transpose()))+(new_x2.dot(new_x2.transpose())))
        return float(sum) / denom
    except ZeroDivisionError:
        return 0


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
        return float(vec-minVal)/(maxVal-minVal)+0.01
    elif maxVal==minVal:
        return vec/maxVal
    else:
        print 'error... maximum value is less than minimum value.'
        raise ArithmeticError

def sigmoid(val):
    return 1/(1+exp(-val))


def denormalize(vec,maxVal,minVal):
    return minVal+(vec-0.01)*(maxVal-minVal)
