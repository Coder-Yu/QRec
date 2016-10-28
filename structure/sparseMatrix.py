from scipy.sparse import csr_matrix
import numpy as np


class SparseMatrix():
    'matrix used to store raw data'
    def __init__(self,data,indices,indptr,shape=None):
        self.matrix = csr_matrix((data,indices,indptr),shape)
        self.shape = self.matrix.shape

    def row(self,r):
        if r >= self.shape[0]:
            return np.zeros((1,self.shape[1]))
        return self.matrix.getrow(r).toarray()
    def col(self,c):
        if c >= self.shape[1]:
            return np.zeros((1, self.shape[0]))
        return self.matrix.getcol(c).toarray().transpose()
    def elem(self,r,c):
        if r >= self.shape[0] or c >= self.shape[1]:
            return 0
        return self.matrix.getrow(r).toarray()[0][c]
    def sRow(self,r):
        'return the sparse row'
        return self.matrix.getrow(r)
    def sCol(self,c):
        'return the sparse column'
        return self.matrix.getcol(c)
    def toDense(self):
        return self.matrix.todense()


