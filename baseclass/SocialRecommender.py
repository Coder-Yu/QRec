from baseclass.IterativeRecommender import IterativeRecommender
from data.social import SocialDAO
from tool import config
class SocialRecommender(IterativeRecommender):
    def __init__(self,conf):
        super(SocialRecommender, self).__init__(conf)
        self.sao = SocialDAO(self.config) #social relations access control

    def readConfiguration(self):
        super(SocialRecommender, self).readConfiguration()
        regular = config.LineConfig(self.config['reg.lambda'])
        self.regS = float(regular['-s'])
