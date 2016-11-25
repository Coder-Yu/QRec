from chart import Chart
from tool.config import Config
from data.rating import RatingDAO
from data.social import SocialDAO

class Display(object):
    def __init__(self,conf):
        if not conf.contains('ratings') and not conf.contains('social'):
            print 'The config file is not in the correct format!'
            exit(-1)
        if conf.contains('ratings'):
            self.dao = RatingDAO(conf)
        if conf.contains('social'):
            self.sao = SocialDAO(conf)


    def draw(self):
        print 'draw chart...'
        #rating
        y = [triple[2] for triple in self.dao.trainingData]
        x = self.dao.rScale
        Chart.hist(x,y,len(self.dao.rScale),'#058edc',
                   'Rating Histogram','Rating Scale','Count')
        y = [len(self.dao.userRated(u)[0]) for u in self.dao.user]
        Chart.distribution(y,'Rating Distribution','','Rated items count per user')
        y = [len(self.dao.itemRated(i)[0]) for i in self.dao.item]
        Chart.distribution(y,'Rating Distribution','','Rated items count per user')
        #social
        x = [len(self.sao.getFollowers(u)) for u in self.sao.user]
        y = [len(self.sao.getFollowees(u)) for u in self.sao.user]
        Chart.scatter(x,y,'red','Follower&Followee','Follower count','Followee count')