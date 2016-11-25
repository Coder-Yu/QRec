from chart import Chart
from tool.config import Config
from data.rating import RatingDAO
from data.social import SocialDAO
from tool.file import FileIO
from tool.qmath import denormalize
from os.path import abspath
import webbrowser

class Display(object):
    def __init__(self,conf):
        self.conf = conf
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
        if self.conf.contains('ratings'):
            y = [triple[2] for triple in self.dao.trainingData]
            x = self.dao.rScale
            if len(x) <20:
                Chart.hist(x,y,len(self.dao.rScale),'#058edc',
                       'Rating Histogram','Rating Scale','Count','../visual/visualization/images/rh')
            y = [len(self.dao.userRated(u)[0]) for u in self.dao.user]
            Chart.distribution(y,'Rating Count Distribution','',
                               'Rated items count per user','../visual/visualization/images/rcu')
            y = [len(self.dao.itemRated(i)[0]) for i in self.dao.item]
            Chart.distribution(y,'Rating Count Distribution','',
                               'user Rated count per item','../visual/visualization/images/rci')

        #social
        if self.conf.contains('social'):
            x = [len(self.sao.getFollowers(u)) for u in self.sao.user]
            y = [len(self.sao.getFollowees(u)) for u in self.sao.user]
            Chart.scatter(x,y,'red','Follower&Followee',
                          'Follower count','Followee count','../visual/visualization/images/ff')
            y = [len(self.sao.getFollowers(u)) for u in self.sao.user]
            Chart.distribution(y, 'Followers Distribution', '',
                               'Followers count per user','../visual/visualization/images/fd1')
            y = [len(self.sao.getFollowees(u)) for u in self.sao.user]
            Chart.distribution(y, 'Followees Distribution', '',
                           'Followees count per user','../visual/visualization/images/fd2')

    def render(self):
        self.draw()
        html ="<html><head><title>Data Analysis</title>\n" \
              "<link rel='stylesheet' type='text/css' href='reportStyle.css'/></head>\n" \
              "<body><div class='reportTitle'><div class='in'>Data Analysis</div></div>\n" \
              "<div class='main'><div class='area1'>\n" \
              "<div class='title'><h3>Data Files</h3></div><div class='text'>"
        if self.conf.contains('ratings'):
            html+="Rating Data: {rating}".format(rating = abspath(self.conf['ratings']))
        if self.conf.contains('social'):
            html+="<br>Social Data: {social}".format(social = abspath(self.conf['social']))
        html+="</div></div><div style='padding-top:20px'><center>" \
              "<img src='images/header2.png'/></center></div>\n"
        if self.conf.contains('ratings'):
            html+="<div class='area1'><div class='title'><h3>Rating Data</h3></div>\n"
            html += "<div class='text'>Rating Scale: {scale}</br>".format(scale=' '.join([str(item) for item in self.dao.rScale]))
            html += "User Count: {user}<br>Item Count: {item}<br>Record Count: {record}<br>Global Mean: {mean}</div>\n"\
                .format(user = str(len(self.dao.user)),item=str(len(self.dao.item)),record = str(len(self.dao.trainingData)),
                        mean = str(round(denormalize(self.dao.globalMean,self.dao.rScale[-1],self.dao.rScale[0]),3)))
            html+="<center><div class='img'><img src='images/rh.png' width='640px' height='480px'/></div></center>\n"
            html += "<center><div class='img'><img src='images/rcu.png' width='640px' height='480px'/></div></center>\n"
            html += "<center><div class='img'><img src='images/rci.png' width='640px' height='480px'/></div></center>\n"
            html += "</div><div style='padding-top:20px'><center>" \
              "<img src='images/header2.png'/></center></div>\n"
        if self.conf.contains('social'):
            html += "<div class='area1'><div class='title'><h3>Social Data</h3></div>\n"
            html += "<div class='text'>User Count: {user}<br>Relation Count: {relation}<br></div>\n" \
                .format(user=str(len(self.sao.user)), relation=str(len(self.sao.relation)))
            html += "<center><div class='img'><img src='images/ff.png' width='640px' height='480px'/></div></center>\n"
            html += "<center><div class='img'><img src='images/fd1.png' width='640px' height='480px'/></div></center>\n"
            html += "<center><div class='img'><img src='images/fd2.png' width='640px' height='480px'/></div></center>\n"
            html += "</div><div style='padding-top:20px'><center>" \
                    "<img src='images/header2.png'/></center></div>\n"

        html+="</div></body></html>"
        FileIO.writeFile('../visual/visualization/','analysis.html',html)
        print 'The report has been output to',abspath('../visual/visualization/analysis.html')
        webbrowser.open(abspath('../visual/visualization/analysis.html'), new=0, autoraise=True)

