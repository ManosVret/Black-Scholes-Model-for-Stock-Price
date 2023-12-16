import numpy as np
import matplotlib.pyplot as plt

#constants
S0 = 50
mu = 0.1
sig0 = 0.2
ksi0 = 0.2


#generate universal W arrays for given dt
def unW(dt):
  length = int(1/dt + 1)
  W1=np.random.normal(0,np.sqrt(dt),length)
  W2=np.random.normal(0,np.sqrt(dt),length)
  W1[0] = 0.0
  W2[0] = 0.0
  

  for i in range(1,len(W1)):
    W1[i] = W1[i] + W1[i-1]
    W2[i] = W2[i] + W2[i-1]
  return W1,W2




def run(p,a,dt, Ws=None): 
  
  T = np.arange(0,1+dt,dt)
  N = len(T)

  if Ws is not None: 
    
    if len(Ws[0]) == N:
      dw1 = np.zeros(N)
      dw2 = np.zeros(N)
      dw1[0] = Ws[0][0]
      dw2[0] = Ws[1][0]
      for i in range(0,len(Ws[1])):
        dw1[i] = Ws[0][i] - Ws[0][i-1]
        dw2[i] = Ws[1][i] - Ws[1][i-1]
    

    
    else:
      fac = int((len(Ws[0]-1))/(N-1))
      fil1 = Ws[0][fac-1::fac]
      fil2 = Ws[1][fac-1::fac]
      dw1 = np.zeros(N)
      dw2 = np.zeros(N)
      dw1[0] = fil1[0]
      dw2[0] = fil2[0]
      for i in range(1,len(fil1)):
        dw1[i] = fil1[i] - fil1[i-1]
        dw2[i] = fil2[i] - fil2[i-1]
    dw1[0] = 0.
  else:
    dw1 =  np.random.normal(0,np.sqrt(dt),N)
    dw2 =  np.random.normal(0,np.sqrt(dt),N)
    dw1[0] = 0.
    dw2[0] = 0.
  ksie = np.zeros(N)
  sige = np.zeros(N)
  ksim = np.zeros(N)
  sigm = np.zeros(N)
  Sm = np.zeros(N)
  Se = np.zeros(N)
  ksie[0] = ksi0
  sige[0] = sig0
  ksim[0] = ksi0
  sigm[0] = sig0
  Sm[0] = S0
  Se[0] = S0
  for i in range(1,N):
    sige[i] = sige[i-1] -(sige[i-1] - ksie[i-1])*dt + p*sige[i-1]*dw2[i]
    sigm[i] = sigm[i-1] -(sigm[i-1] - ksim[i-1])*dt + p*sigm[i-1]*dw2[i] + 0.5*(p**2)*sigm[i-1]*(dw2[i]**2-dt)
    ksie[i] = ksie[i-1] + 1/a * (sige[i-1] - ksie[i-1])*dt
    ksim[i] = ksim[i-1] + 1/a * (sigm[i-1] - ksim[i-1])*dt
    Se[i] = Se[i-1] +  mu*Se[i-1]*dt + Se[i-1]*sige[i-1]*dw1[i]
    Sm[i] = Sm[i-1] +  mu*Sm[i-1]*dt + Sm[i-1]*sigm[i-1]*dw1[i] + Sm[i-1]*0.5*(sigm[i-1]**2)*((dw1[i]**2)-dt)
    '''
    sige[i] = sige[i-1] -(sige[i-1] - ksie[i-1])*dt + p*sige[i-1]*dw1[i]

    sigm[i] = sigm[i-1] -(sigm[i-1] - ksim[i-1])*dt + p*sigm[i-1]*dw1[i] + 0.5*(p**2)*sigm[i-1]*(dw1[i]**2-dt)

    ksie[i] = ksie[i-1] + 1/a * (sige[i-1] - ksie[i-1])*dt

    ksim[i] = ksim[i-1] + 1/a * (sigm[i-1] - ksim[i-1])*dt


    Se[i] = Se[i-1] +  mu*Se[i-1]*dt + Se[i-1]*sige[i-1]*dw1[i]

    Sm[i] = Sm[i-1] +  mu*Sm[i-1]*dt + Sm[i-1]*sigm[i-1]*dw1[i] + Sm[i-1]*0.5*(sigm[i-1]**2)*((dw1[i]**2)-dt)
    '''

  return T, Se, Sm, ksie, sige, ksim, sigm, dw1 ,dw2


def runbs(dt,Ws): #cannot call without defining universal Ws
  T = np.arange(0,1+dt,dt)
  N = len(T)

  if len(Ws[0]) == N:
    dw1 = np.zeros(N)
    dw1[0] = Ws[0][0]
    for i in range(1,len(Ws[1])):
      dw1[i] = Ws[0][i] - Ws[0][i-1]

  
  else: #in order to recalculate dW if \
  # realisations need to be run for different  \ 
  # dts using same stochastic array
    fac = int((len(Ws[0]-1))/(N-1))
    fil1 = Ws[0][fac-1::fac]
    dw1 = np.zeros(N)
    dw1[0] = fil1[0]
    for i in range(1,len(fil1)):
      dw1[i] = fil1[i] - fil1[i-1]

  Sm = np.zeros(N)
  Se = np.zeros(N)
  Sm[0] = S0
  Se[0] = S0

  for i in range(1,N):
    Se[i] = Se[i-1] +  mu*Se[i-1]*dt + Se[i-1]*sig0*dw1[i]

    Sm[i] = Sm[i-1] +  mu*Sm[i-1]*dt + Sm[i-1]*sig0*dw1[i] + Sm[i-1]*0.5*(sig0**2)*((dw1[i]**2)-dt)

  return T, Se, Sm