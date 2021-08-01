import numpy as np
#class Triple(object):


class SparseMatrix():
    'matrix used to store raw data'
    def __init__(self,triple):
        self.matrix_User = {}
        self.matrix_Item = {}
        for item in triple:
            if item[0] not in self.matrix_User:
                self.matrix_User[item[0]] = {}
            if item[1] not in self.matrix_Item:
                self.matrix_Item[item[1]] = {}
            self.matrix_User[item[0]][item[1]]=item[2]
            self.matrix_Item[item[1]][item[0]]=item[2]
        self.elemNum = len(triple)
        self.size = (len(self.matrix_User),len(self.matrix_Item))

    def sRow(self,r):
        if r not in self.matrix_User:
            return {}
        else:
            return self.matrix_User[r]

    def sCol(self,c):
        if c not in self.matrix_Item:
            return {}
        else:
            return self.matrix_Item[c]

    def row(self,r):
        if r not in self.matrix_User:
            return np.zeros((1,self.size[1]))
        else:
            array = np.zeros((1,self.size[1]))
            ind = list(self.matrix_User[r].keys())
            val = list(self.matrix_User[r].values())
            array[0][ind] = val
            return array

    def col(self,c):
        if c not in self.matrix_Item:
            return np.zeros((1,self.size[0]))
        else:
            array = np.zeros((1,self.size[0]))
            ind = list(self.matrix_Item[c].keys())
            val = list(self.matrix_Item[c].values())
            array[0][ind] = val
            return array
    def elem(self,r,c):
        if not self.contains(r,c):
            return 0
        return self.matrix_User[r][c]

    def contains(self,r,c):
        if r in self.matrix_User and c in self.matrix_User[r]:
            return True
        return False

    def elemCount(self):
        return self.elemNum

    def size(self):
        return self.size


