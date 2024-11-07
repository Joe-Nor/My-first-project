import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
'''连续改变箱的大小L，算得的基态能与真实值比较'''
# 定义矩阵的大小
N = 100 # 设定矩阵大小为 N x N
j = 1  # 定义第几个能级
L = np.linspace(10, 300, 100)
V = 1 # 势阱深
a = 1 # 势阱宽
E = np.zeros((len(L)))
for j in range(len(L)):
          
          H = np.zeros((N, N))
          f = np.zeros((N))
          k = np.zeros((N))

          for i in range(N):
            k[i] =  i * np.pi / L[j] 

          '''print("矩阵 k:")
          print(k)'''

          for m in range(N):
            for n in range(N):
                    if m == n: 
                      if n != 0 :
                         H[m, n] =  k[m]**2 - V/L[j]*(a+np.sin(2*k[n]*a)/(2*k[n]))
                      else:
                         H[m, n] =  k[m]**2 - V/L[j]*a
                    else:  
                         H[m, n] = - V/L[j] * (np.sin((k[m]-k[n])*a)/(k[m]-k[n])+np.sin((k[m]+k[n])*a)/(k[m]+k[n]))

          '''print("矩阵 H:")
          print(H)'''

          eigvals, eigvecs = eigh(H)
          E[j] = eigvals[0]


print("能级：")
print(E)
plt.plot(L,E)
plt.plot(L,np.full(len(L), -0.4538))
plt.show()

