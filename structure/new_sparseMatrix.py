import numpy as np


class SparseMatrix():
    'matrix used to store raw data'
    def __init__(self,triple,shape):
        self.matrix_User = {}
        self.matrix_Item = {}
        self.shape = shape
        for item in triple:
            if not self.matrix_User.has_key(item[0]):
                self.matrix_User[item[0]] = {}
            if not self.matrix_Item.has_key(item[1]):
                self.matrix_Item[item[1]] = {}
            self.matrix_User[item[0]][item[1]]=item[2]
            self.matrix_Item[item[1]][item[0]]=item[2]
        self.elemNum = len(triple)

    def row(self,r):
        if not self.matrix_User.has_key(r):
            return np.zeros((1,self.shape[1]))
        else:
            array = np.zeros((1,self.shape[1]))
            ind = self.matrix_User[r].keys()
            val = self.matrix_User[r].values()
            array[0][ind] = val
            return array

    def col(self,c):
        if not self.matrix_Item.has_key(c):
            return np.zeros((1,self.shape[0]))
        else:
            array = np.zeros((1,self.shape[0]))
            ind = self.matrix_Item[c].keys()
            val = self.matrix_Item[c].values()
            array[0][ind] = val
            return array
    def elem(self,r,c):
        if not self.matrix_User.has_key(r) or not self.matrix_User[r].has_key(c):
            return 0
        return self.matrix_User[r][c]
    def elemCount(self):
        return self.elemNum
    # def sRow(self,r):
    #     'return the sparse row'
    #     return self.matrix.getrow(r)
    # def sCol(self,c):
    #     'return the sparse column'
    #     return self.matrix.getcol(c)
    # def toDense(self):
    #     return self.matrix.todense()


