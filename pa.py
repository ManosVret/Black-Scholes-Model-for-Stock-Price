import numpy as np
import matplotlib.pyplot as plt
from funcs import *


def effPAonks(): #effects of p and a on ksi and sigma

    pl = [0.1,0.3,0.5,0.7,1]

    al = [0.01,0.05,0.1,0.5,1]



    plt.style.use('seaborn-dark-palette')
    fig,axs = plt.subplots(len(al),len(pl),sharex=True,sharey=True)

    for i in range(len(al)):
      for j in range(len(pl)):
        
        axis = axs[i,j]
        np.random.seed(75)
        title = r'$p=$' + str(pl[j]) + '  ' + r'$\alpha=$' + str(al[i])
    ##    name = 'p'+str(pl[j]).replace('.','')+'a'+str(al[i]).replace('.','')+'.pdf'
        data = run(pl[j],al[i],0.01)
    ##    plt.suptitle(title)
        axis.title.set_text(title)
        axis.plot(data[0],data[3],label=r'$\xi$',color = 'indianred')
        axis.plot(data[0],data[4],label=r'$\sigma$', color = 'cornflowerblue')
    ##    axis.plot(data[0],data[1])
        axis.grid()
        
    ##    plt.legend()
    ##    plt.show()
      ##  plt.savefig(fname=name)


    fig.text(0.5, 0.04, r'$t [yrs]$', ha='center')
    ##fig.text(0.04, 0.5, r'$S [€]$', va='center', rotation='vertical')

    handles, labels = axs[i,j].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left')

    plt.show()
    return 


def effPAonS():
    pl = [0.1,2]
    al = [0.1,1]
    dt = 0.001

    devs = np.zeros((len(pl),len(al)))
    N = len(np.arange(0,1+dt,dt))


    plt.style.use('seaborn-dark-palette')
    fig,axs = plt.subplots(len(al),len(pl),sharex=True,sharey=True)

    for i in range(len(al)):
      for j in range(len(pl)):
        
        # np.random.seed(463)
        np.random.seed(11)
        
        title = r'$p=$' + str(pl[j]) + '  ' + r'$\alpha=$' + str(al[i])
        for k in range(3):
          dat = run(pl[i],al[i],dt)
          axs[i,j].plot(dat[0],dat[2])
        
        axs[i,j].grid()
        axs[i,j].title.set_text(title)
        ''''
        reals= 100
        Ss = np.zeros((reals,N))

        title = r'$p=$' + str(pl[j]) + '  ' + r'$\alpha=$' + str(al[i])
        
        for k in range(reals):
          dat = run(pl[j],al[i],0.01)
          Ss[i] = dat[1]
          
        Ssd = np.std(Ss,axis=0)
        axs[i,j].plot(dat[0],Ssd)
        axs[i,j].grid()
        axs[i,j].title.set_text(title)
        '''
    fig.text(0.5, 0.04, r'$t [yrs]$', ha='center')
    fig.text(0.04, 0.5, r'$S [€]$', va='center', rotation='vertical')
    plt.show()

    return