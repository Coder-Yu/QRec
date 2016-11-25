import numpy as np
from structure import sparseMatrix,new_sparseMatrix
from tool.config import Config,LineConfig
from tool.qmath import normalize
import os.path
from re import split

class SocialDAO(object):
    def __init__(self,conf):
        self.config = conf
        self.socialConfig = LineConfig(self.config['social.setup'])
        self.user = {} #used to store the order of users
        self.relation = []
        self.followees = {}
        self.followers = {}
        self.trustMatrix = self.loadRelationship(self.config['social'])


    def loadRelationship(self,filePath):
        print 'load social data...'
        triple = []
        with open(filePath) as f:
            relations = f.readlines()
            # ignore the headline
        if self.socialConfig.contains('-header'):
            relations = relations[1:]
        # order of the columns
        order = self.socialConfig['-columns'].strip().split()
        if len(order)<=2:
            print 'The social file is not in a correct format.'
        for line in relations:
            items = split(' |,|\t', line.strip())
            if len(order) < 2:
                print 'The social file is not in a correct format. Error: Line num %d' % lineNo
                exit(-1)
            userId1 = items[int(order[0])]
            userId2 = items[int(order[1])]
            if len(order)<3:
                weight = 1
            else:
                weight = float(items[int(order[2])])
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
            self.relation.append([userId1,userId2,weight])
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
