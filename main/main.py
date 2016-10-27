from algorithm.rating.UserKNN import UserKNN
from tool.config import Config
c = Config('../config/UserKNN')
algor = UserKNN(c)

algor.execute()
