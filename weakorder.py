import numpy as np
import matplotlib.pyplot as plt
from funcs import *



def WObs(): #weak order convergence - black scholes
    

    def h(x): #define funciton h(x)
      return x**2 + 2*x


    dts = np.array([0.01,0.02,0.05,0.08])
    dts = dts/10
    dts = np.sort(dts)

    errore = []
    errorm = []

    N=1000

    for dt in dts:
      print(dt)
      avge = 0.
      avgm = 0.
      avga = 0.
      totala = 0.
      totale = 0.
      totalm = 0.
      for i in range(N):
        Ws = unW(dt)
        data = runbs(dt,Ws)
          
        totala = totala + h(S0*np.exp((mu-0.5*sig0**2)*data[0][-2] + sig0*(Ws[0][-1])))
        totale = totale + h(data[1][-1])
        totalm = totalm + h(data[2][-1])
        

      avge = totale/(i+1)
      avgm = totalm/(i+1)
      avga = totala/(i+1)
      print('a=',avga)
      print('e=',avge)
      print('m=',avgm)
      errore.append(abs(avga-avge))
      errorm.append(abs(avga-avgm))

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
    print('weak order of convergence of euler=',grade)
    print('weak order of convergence of milstein=',gradm)


    handles, labels = axs[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left')
    plt.show()



'''2.2.2 full model'''
def WOfull(): #weak order convergence - full model
    dts = np.array([0.02, 0.04, 0.05, 0.08,0.1])
    dts = dts/100
    dta = 0.01/1000

    N = 1000
    mate = np.zeros((N,5))
    matm = np.zeros((N,5))
    matea = np.zeros((N,5))
    matma = np.zeros((N,5))


    for i in range(N):
      es = []
      ms = []
      aes = []
      ams = []
      Ws = unW(dta)
      anal = run(0.1,1,dta,Ws)
      for dt in dts:
    ##    print(dt)
        data = run(0.1,1,dt,Ws)
        es.append(data[1][-1])
        ms.append(data[2][-1])
        aes.append(anal[1][-1])
        ams.append(anal[2][-1])
    ##  plt.plot(dts,errore)
    ##  plt.plot(dts,errorm)
      mate[i,:] = es
      matm[i,:] = ms
      matea[i,:] = aes
      matma[i,:] = ams

    errore = np.absolute(np.mean(mate,axis=0) - np.mean(matea,axis=0))
    errorm = np.absolute(np.mean(matm,axis=0) - np.mean(matma,axis=0))
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
    print('weak order of convergence of euler=',grade)
    print('weak order of convergence of milstein=',gradm)


    handles, labels = axs[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left')
    plt.show()
    return