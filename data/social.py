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
        self.triple = []
        self.trustMatrix = self.loadRelationship(self.config['social'])


    def loadRelationship(self,filePath):
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
                weight = items[int(order[2])]
            # order the user
            if not self.user.has_key(userId1):
                self.user[userId1] = len(self.user)
            if not self.user.has_key(userId2):
                self.user[userId2] = len(self.user)
            self.triple.append([self.user[userId1], self.user[userId2], float(weight)])
        return new_sparseMatrix.SparseMatrix(self.triple)

    def row(self,u):
        #return user u's followees
        return self.trustMatrix.row(self.user[u])

    def col(self,u):
        #return user u's followers
        return self.trustMatrix.col(self.user[u])

    def weight(self,u1,u2):
        return self.trustMatrix.elem(self.user[u1],self.user[u2])

    def trustSize(self):
        return self.trustMatrix.size

    def followers(self,u):
        if self.user.has_key(u):
            return self.trustMatrix.matrix_Item[self.user[u]]
        else:
            return {}

    def followees(self,u):
        if self.user.has_key(u):
            return self.trustMatrix.matrix_User[self.user[u]]
        else:
            return {}

    def hasFollowee(self,u1,u2):
        if self.user.has_key(u1):
            if self.trustMatrix.matrix_User[self.user[u1]].has_key(self.user[u2]):
                return True
            else:
                return False
        return False

    def hasFollower(self,u1,u2):
        if self.user.has_key(u1):
            if self.trustMatrix.matrix_Item[self.user[u1]].has_key(self.user[u2]):
                return True
            else:
                return False
        return False
