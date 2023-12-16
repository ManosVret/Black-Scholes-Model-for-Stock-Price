import numpy as np
import matplotlib.pyplot as plt
from funcs import *


def SObs(): #strong order convergence of black-scholes model

    dts = np.array([0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08])
    dts = dts/10
    dts = np.sort(dts)

    errorm = []
    errore = []

    for dt in dts:
      print(dt)
    ##  Ws = unW(dt)
    ##  data = runbs(dt,Ws)
    ##  anal = S0*np.exp((mu-0.5*sig0**2)*1 + sig0*(Ws[0][-1]))
    ##
      totale = 0.
      totalm = 0.
      
      
      for i in range(10000):
        Ws = unW(dt)
        data = runbs(dt,Ws)
          
        anal = S0*np.exp((mu-0.5*sig0**2)*data[0][-2] + sig0*(Ws[0][-1]))
    ##      print(data[0][-2])
        totale = totale +abs(anal-data[1][-1])
        totalm = totalm +abs(anal-data[2][-1])

      avge = totale / (i+1)
      avgm = totalm / (i+1)
      errorm.append(avgm)
      errore.append(avge)
      
    ##plt.plot(dts,errorm, label='milstein')
    ##plt.plot(dts,errore, label='euler')
    ##plt.legend()
    ##plt.grid()
    ##plt.show()


    plt.style.use('seaborn-dark-palette')
    adts = np.log10(dts)
    aerrorm = np.log10(np.asarray(errorm))
    aerrore = np.log10(np.asarray(errore))

    grade = np.mean(np.diff(aerrore)/np.diff(adts))
    gradm = np.mean(np.diff(aerrorm)/np.diff(adts))
    print('strong order of convergence of euler=',grade)
    print('strong order of convergence of milstein=',gradm)


    fig, axs = plt.subplots(1,2)
    axs[0].plot(adts,aerrorm,label='milstein')
    axs[0].plot(adts,aerrore,label='euler')
    axs[0].set_ylabel('log(error)')
    axs[0].set_xlabel('log(dt)')
    axs[0].grid()
    axs[1].plot(dts,errorm,label='milstein')
    axs[1].plot(dts,errore,label='euler')
    axs[1].set_ylabel('error')
    axs[1].set_xlabel('dt')
    axs[1].grid()


    handles, labels = axs[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left')
    plt.show()
    return
            


def SOfull(): #strong order convergence of full model
    dts = np.array([0.01,0.02, 0.04,0.05])
    dts = dts/10

    errore = []
    errorm = []

    for dt in dts:
     print(dt)
     totale = 0.
     totalm = 0.

     for i in range(1000):
       dta = min(dts)/10
       Ws = unW(dta)
       anal = run(1,10,dta,Ws)
       data = run(1,10,dt,Ws)
       totale = totale + abs(anal[1][-1]-data[1][-1])
       totalm = totalm + abs(anal[2][-1]-data[2][-1])

     errore.append(totale/(i+1))
     errorm.append(totalm/(i+1))



    plt.style.use('seaborn-dark-palette')

    adts = np.log10(dts)
    aerrorm = np.log10(np.asarray(errorm))
    aerrore = np.log10(np.asarray(errore))

    fig, axs = plt.subplots(1,2)
    axs[0].plot(adts,aerrorm,label='milstein')
    axs[0].plot(adts,aerrore,label='euler')
    axs[0].set_ylabel('log(error)')
    axs[0].set_xlabel('log(dt)')
    axs[0].grid()
    axs[1].plot(dts,errorm,label='milstein')
    axs[1].plot(dts,errore,label='euler')
    axs[1].set_ylabel('error')
    axs[1].set_xlabel('dt')
    axs[1].grid()

    grade = np.mean(np.diff(aerrore)/np.diff(adts))
    gradm = np.mean(np.diff(aerrorm)/np.diff(adts))
    print('strong order of convergence of euler=',grade)
    print('strong order of convergence of milstein=',gradm)


    handles, labels = axs[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left')
    plt.show()
    return