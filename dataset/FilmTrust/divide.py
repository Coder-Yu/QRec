test = []
train = []
import random
with open('ratings.txt') as f:
    for line in f:
        #items= line.strip().split()
        if random.random()>0.05:
            train.append(line)
        else:
            test.append(line)

with open('testset.txt','w') as f:
    f.writelines(test)
with open('trainset.txt','w') as f:
    f.writelines(train)