import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
'''对于很大的L，连续改变N，最终趋于稳定'''
# 定义矩阵的大小
Nmax = 1000 # 设定矩阵大小为 N x N
j = 1  # 定义第几个能级
L = 1000 # 周期性边界
V = 1 # 势阱深
a = 1 # 势阱宽
N = list(range(100, 1001, 10))
print(N)
E = np.zeros((len(N)))
for j in range(len(N)):
          H = np.zeros((N[j], N[j]))
          f = np.zeros((N[j]))
          k = np.zeros((N[j]))

          for i in range(N[j]):
            k[i] =  i * np.pi / L 


          for m in range(N[j]):
            for n in range(N[j]):
                    if m == n: 
                      if n != 0 :
                         H[m, n] =  k[m]**2 - V/L*(a+np.sin(2*k[n]*a)/(2*k[n]))
                      else:
                         H[m, n] =  k[m]**2 - V/L*a
                    else:  
                     H[m, n] = - V/L * (np.sin((k[m]-k[n])*a)/(k[m]-k[n])+np.sin((k[m]+k[n])*a)/(k[m]+k[n]))

          '''print("矩阵 H:")
          print(H)'''

          eigvals, eigvecs = eigh(H)
          E[j] = eigvals[0]

print("能级：")
print(E)
plt.plot(N,E)
plt.show()
