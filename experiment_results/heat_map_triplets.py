import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_data(fname):
    results = np.loadtxt(fname, delimiter=',')
    if results[len(results)-1, 0] == 1.0:
        r = [results[len(results)-1, i] for i in range(1,6)]
        r.insert(0,1)
        r.insert(len(r), results[len(results)-2, 4])
        print (results[len(results)-2, 4])
        return r
    else:
        return [0 for _ in range(7)]

parameter = []

step = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
item = [10, 20, 40, 80]

step_dict={'0.1':0 , '0.2':1, '0.3':2, '0.4':3, '0.5':4, '0.6':5, '0.7':6, '0.8':7, '0.9':8}
item_dict = {'10':0, '20':1, '40':2, '80':3}

proportion_success = np.zeros([len(step), len(item)])
time = np.zeros([len(step), len(item)])
stopped = np.zeros([len(step), len(item)])

for i in item:
    for s in step:
        for experiment in range(100):
            parameter.append([i,s,experiment])

for param in parameter:
    fname = str(param[0])  + '/' + 'triplet_result_' + str(param[0]) + '_' + str(param[1]) + '_' + str(param[2])  + '.txt'
    result = get_data(fname)
    proportion_success[int(step_dict[str(param[1])]),int(item_dict[str(param[0])])] += result[0]

    if result[6]==0.0 and result[0]==1:
        time[int(step_dict[str(param[1])]),int(item_dict[str(param[0])])] += 1
    #print(param)

time = np.divide(time, proportion_success)
proportion_success = proportion_success / 100

print(proportion_success)
print(time)
print(stopped)

import matplotlib.cm
fig, ax = plt.subplots()

plt.imshow(time, interpolation='none',  origin='lower', cmap=matplotlib.cm.coolwarm)
plt.title('Proportion of Times Final Gradient = 0 \n over Successful Embeddings')
#labels_x = ['40', '80']
labels_x = ['10', '20', '40', '80']
labels_y = ['.1', '.2', '.3', '.4', '.5', '.6', '.7', '.8', '.9']
ax.set_xticklabels(labels_x)
ax.set_yticklabels(labels_y)
ax.set_xticks(np.arange(len(labels_x)))
ax.set_yticks(np.arange(len(labels_y)))
# plt.xticks(labels_x)
# plt.yticks(labels_y)
plt.colorbar()
plt.xlabel('items')
plt.ylabel('decay rate')
plt.show()
