#A snippet showing how to directly run algorithms in QRec
from main.QRec import QRec #need to be modified according to your path
from tool.config import Config #need to be modified according to your path
#create your own config file
#using absolute path for both config file and dataset files
config = Config("/home/xxx/user.conf")
rec = QRec(config)
rec.execute()