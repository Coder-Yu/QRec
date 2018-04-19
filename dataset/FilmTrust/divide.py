test = []
train = []
with open('ratings.txt') as f:
    import random
    for line in f:
        items= line.strip().split()
        if items[-1]<'3':
            train.append(items[0]+' '+items[1]+' 1\n')

with open('testset.txt','w') as f:
    f.writelines(test)
with open('sratings.txt','w') as f:
    f.writelines(train)