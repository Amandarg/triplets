parameter = {}

step = {.1, .2, .3, .4, .5, .6, .7, .8, .9}
item = {10, 20, 40, 80, 160}

count = 1
for i in item:
    for s in step:
        parameter[count]=[i, s]
        count +=1

print(parameter)
