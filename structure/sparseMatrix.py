from scipy.sparse import csr_matrix
import numpy as np


class SparseMatrix():
    'matrix used to store raw data'
    def __init__(self,data,indices,indptr,shape=None):
        self.matrix = csr_matrix((data,indices,indptr),shape)

    def row(self,r):
        return self.matrix.getrow(r).toarray()
    def col(self,c):
        return self.matrix.getcol(c).toarray().transpose()
    def sRow(self,r):
        'return the sparse row'
        return self.matrix.getrow(r)
    def sCol(self,c):
        'return the sparse column'
        return self.matrix.getcol(c)
    def toDense(self):
        return self.matrix.todense()


