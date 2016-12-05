from baseclass.IterativeRecommender import IterativeRecommender
from data.social import SocialDAO
from tool import config
class SocialRecommender(IterativeRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(SocialRecommender, self).__init__(conf,trainingSet,testSet,fold)
        self.sao = SocialDAO(self.config) #social relations access control

    def readConfiguration(self):
        super(SocialRecommender, self).readConfiguration()
        regular = config.LineConfig(self.config['reg.lambda'])
        self.regS = float(regular['-s'])
