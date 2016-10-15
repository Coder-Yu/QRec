import numpy as np
import os.path
class DAO(object):
    'data access control'
    def __init__(self,config):
        self.config = config
        self.user = {}
        self.item = {}
        self.rating = {}
        self.timestamp = {}

    def loadRatings(self):
        if not os.path.exists(self.config['dataset.ratings']):
            print 'ratings file is not found!'
            exit()
        with open(self.config['dataset.ratings']) as f:
            f.readlines()


