import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
'''有限深势阱，以平面波为基，在某些情况下，计算的基态能会比“真实值”要低'''
# 定义矩阵的大小
N = 100 # 设定矩阵大小为 N x N
j = 1  # 定义第几个能级
L = 10 # 周期性边界
V = 1 # 势阱深
a = 1 # 势阱宽

H = np.zeros((N, N))
f = np.zeros((N))
k = np.zeros((N))

for i in range(N):
   k[i] =  i * np.pi / L 

'''print("矩阵 k:")
print(k)'''

for m in range(N):
    for n in range(N):
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

# 输出特征值和特征向量
print("特征值：")
print(eigvals)
'''print("特征向量：")
print(eigvecs)'''

'''v_normalized = eigvecs / np.linalg.norm(eigvecs)'''

def f(x):
     result = 0 #初始化函数值为0
     for i in range(N):
         
         result += eigvecs[i,j-1] * (1/np.sqrt(L)) * np.cos(k[i]*x)
        
     return result

# 生成 x 的值范围
x = np.linspace(-L, L, 400)

y = f(x)
if y[200] > 0:
     y=y
else:
     y=-y
   
# 创建图形
plt.plot(x, y, label='calculation')

# 添加标题和标签
plt.title("Plot of calculation")
plt.xlabel("x")
plt.ylabel("y")

# 显示图例
plt.legend()

# 显示图形
plt.show()
