import time
start = time.time()
parameter = []

step = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
item = [10, 20, 40, 80, 160]

for experiment in range(100):
    for i in item:
        for s in step:
            parameter.append([experiment, i, s])
end =time.time()
print(end-start)
