import os.path
class Config(object):
    def __init__(self,fileName):
        self.config = {}
        self.readConfiguration(fileName)


    def readConfiguration(self,fileName):
        path = '../config/'+fileName
        if not os.path.exists(path):
            print 'config file is not found!'
            exit()
        with open(path) as f:
            for line in f:
                key,value=line.strip().split('=')
                self.config[key]=value


