#A snippet showing how to directly run algorithms in QRec
from QRec import Recommender #need to be modified according to your path
from tool.config import Config #need to be modified according to your path
#-----------------------------------------------------------------------------------
#create your own config file by following the file format in the directory of config
#-----------------------------------------------------------------------------------
config = Config("/home/xxx/algor_name.conf")
rec = Recommender(config)
rec.execute()
#-----------------------------------------------------------------------------------

#your own codes to use the recommendation results

#-----------------------------------------------------------------------------------