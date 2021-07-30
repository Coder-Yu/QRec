#A snippet showing how to directly run algorithms in QRec
from main.QRec import QRec
from tool.config import Config
config = Config("UserKNN.conf")
rec = QRec(config)
rec.execute()