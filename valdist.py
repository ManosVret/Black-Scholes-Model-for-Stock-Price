import numpy as np
import matplotlib.pyplot as plt
from funcs import *


def valdist(): #generage distribution of values at T=1
    '''
    pl = [0.1,0.3,0.5,0.7,1]

    al = [0.01,0.05,0.1,0.5,1]

    ##pl = [0.1,2]
    ##al = [0.1,1]


    plt.style.use('seaborn-dark-palette')
    fig,axs = plt.subplots(len(al),len(pl),sharey=True, sharex=True)

    for i in range(len(al)):
      print(i)
      for j in range(len(pl)):
        print(j)
        np.random.seed(743)
        
        
        res= []
        for k in range(1000):
          res.append(run(pl[j],al[i],0.01)[1][-1])
        res = np.around(np.asarray(res),0)
        mi = min(res)
        ma = max(res)
        S = np.arange(mi,ma+1,1)
        dist = np.zeros(len(S))

        for l in range(len(dist)):
          dist[l] = np.sum(res==S[l])
      
        title = r'$p=$' + str(pl[j]) + '  ' + r'$\alpha=$' + str(al[i])
        axs[i,j].grid()
        axs[i,j].title.set_text(title)
        axs[i,j].bar(S,dist)
        axs[i,j].set_xlim([0,100])

    plt.show()
    '''

    res= []
    for k in range(1000):
      res.append(run(1,1,0.01)[1][-1])
    res = np.around(np.asarray(res),0)
    mi = min(res)
    ma = max(res)
    S = np.arange(mi,ma+1,1)
    dist = np.zeros(len(S))

    for l in range(len(dist)):
      dist[l] = np.sum(res==S[l])
    plt.grid()
    plt.bar(S,dist)
    plt.show()
    return