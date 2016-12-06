import numpy as np
from structure import sparseMatrix,new_sparseMatrix
from tool.config import Config,LineConfig
from tool.qmath import normalize
import os.path
from re import split

class SocialDAO(object):
    def __init__(self,conf,relation=list()):
        self.config = conf
        self.user = {} #used to store the order of users
        self.relation = relation
        self.followees = {}
        self.followers = {}
        self.trustMatrix = self.__generateSet()

    def __generateSet(self):
        triple = []
        for line in self.relation:
            userId1,userId2,weight = line
            #add relations to dict
            if not self.followees.has_key(userId1):
                self.followees[userId1] = {}
            self.followees[userId1][userId2] = weight
            if not self.followers.has_key(userId2):
                self.followers[userId2] = {}
            self.followers[userId2][userId1] = weight
            # order the user
            if not self.user.has_key(userId1):
                self.user[userId1] = len(self.user)
            if not self.user.has_key(userId2):
                self.user[userId2] = len(self.user)
            triple.append([self.user[userId1], self.user[userId2], weight])
        return new_sparseMatrix.SparseMatrix(triple)

    def row(self,u):
        #return user u's followees
        return self.trustMatrix.row(self.user[u])

    def col(self,u):
        #return user u's followers
        return self.trustMatrix.col(self.user[u])

    def elem(self,u1,u2):
        return self.trustMatrix.elem(u1,u2)

    def weight(self,u1,u2):
        if self.followees.has_key(u1) and self.followees[u1].has_key(u2):
            return self.followees[u1][u2]
        else:
            return 0

    def trustSize(self):
        return self.trustMatrix.size

    def getFollowers(self,u):
        if self.followers.has_key(u):
            return self.followers[u]
        else:
            return {}

    def getFollowees(self,u):
        if self.followees.has_key(u):
            return self.followees[u]
        else:
            return {}

    def hasFollowee(self,u1,u2):
        if self.followees.has_key(u1):
            if self.followees[u1].has_key(u2):
                return True
            else:
                return False
        return False

    def hasFollower(self,u1,u2):
        if self.followers.has_key(u1):
            if self.followers[u1].has_key(u2):
                return True
            else:
                return False
        return False
