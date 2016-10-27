test = []
train = []
with open('ratings.txt') as f:
    import random
    for line in f:
        if random.random()<0.1:
            test.append(line)
        else:
            train.append(line)

with open('testset.txt','w') as f:
    f.writelines(test)
with open('trainset.txt','w') as f:
    f.writelines(train)