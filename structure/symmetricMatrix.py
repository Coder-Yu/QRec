import numpy as np

class SymmetricMatrix(object):
    def __init__(self, shape):
        self.symMatrix = {}
        self.shape = (shape,shape)

    def __getitem__(self, item):
        if self.symMatrix.has_key(item):
            return self.symMatrix[item]
        return {}

    def set(self,i,j,val):
        if not self.symMatrix.has_key(i):
            self.symMatrix[i] = {}
        self.symMatrix[i][j]=val
        if not self.symMatrix.has_key(j):
            self.symMatrix[j] = {}
        self.symMatrix[j][i] = val


    def get(self,i,j):
        if not self.symMatrix.has_key(i) or not self.symMatrix[i].has_key(j):
            return 0
        return self.symMatrix[i][j]

    def contains(self,i,j):
        if self.symMatrix.has_key(i) and self.symMatrix[i].has_key(j):
            return True
        else:
            return False

    # def row(self, r):
    #     if not self.symMatrix.has_key(r):
    #         return np.zeros((1, self.shape[1]))
    #     else:
    #         array = np.zeros((1, self.shape[1]))
    #         for item in self.symMatrix[r]:
    #             array[0][item] = self.symMatrix[r][item]
    #         return array
    #
    #
    # def col(self, c):
    #     return self.row(c)
    #
    #
    # def elem(self, r, c):
    #     self.get(r,c)
        # def sRow(self,r):
        #     'return the sparse row'
        #     return self.matrix.getrow(r)
        # def sCol(self,c):
        #     'return the sparse column'
        #     return self.matrix.getcol(c)
        # def toDense(self):
        #     return self.matrix.todense()