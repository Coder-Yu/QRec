import os.path
class Config(object):
    def __init__(self,fileName):
        self.config = {}
        self.readConfiguration(fileName)

    def __getitem__(self, item):
        return self.config[item]

    def getOptions(self,item):
        return self.config[item]

    def readConfiguration(self,fileName):
        path = '../config/'+fileName
        if not os.path.exists(path):
            print 'config file is not found!'
            raise IOError
        with open(path) as f:
            for line in f:
                key,value=line.strip().split('=')
                self.config[key]=value



class LineConfig(object):
    def __init__(self,content):
        self.line = content.strip().split(' ')
        self.options = {}
        self.mainOption = False
        if self.line[0] == 'on':
            self.mainOption = True
        elif self.line[0] == 'off':
            self.mainOption = False
        for i,item in enumerate(self.line):
            if (item.startswith('-') or item.startswith('--')) and  not item[1:].isdigit():
                ind = i+1
                for j,sub in enumerate(self.line[i+1:]):
                    if (sub.startswith('-') or sub.startswith('--')) and  not sub[1:].isdigit():
                        ind = j
                        break
                try:
                    self.options[item] = ' '.join(self.line[i+1:i+1+ind])
                except IndexError:
                    self.options[item] = 1


    def __getitem__(self, item):
        return self.options[item]

    def getOption(self,key):
        return self.options[key]

    def isMainOn(self):
        return self.mainOption

    def contains(self,key):
        return self.options.has_key(key)


