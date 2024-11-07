import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
'''氢原子基态能求解，用了固定的基函数，非线性参数已给定'''

N = 4 
a = [13.00773 , 1.962079 , 0.444529 , 0.1219492]


A = np.zeros((N, N))
B = np.zeros((N, N))
f = np.zeros((N))

# 根据给定规则生成矩阵 A 和 B
for m in range(N):
    for n in range(N):
            A[m, n] = 3 * a[m] * a[n] * (np.pi)**1.5 / ((a[m]+a[n])**2.5) - 2 * np.pi/(a[m]+a[n])
            B[m, n] = (np.pi / (a[m]+a[n]))**1.5


# 输出生成的矩阵 A 和 B
print("矩阵 A:")
print(A)

print("矩阵 B:")
print(B)

eigvals, eigvecs = eigh(A, B) 

# 输出特征值和特征向量
print("特征值：")
print(eigvals)
print("特征向量：")
print(eigvecs)

v_normalized = eigvecs[:,0] / np.sqrt(np.dot(np.dot(eigvecs[:,0].T,B), eigvecs[:,0]))
# 求解的波函数与严格解之间的可视化
def f(x):
     result = 0 #初始化函数值为0
     for i in range(N):
         result += v_normalized[i] * np.exp(- a[i]* (x**2) )
        
     return result


def q(x): 
 result = 2 * np.exp(-x)  /np.sqrt(4*np.pi)
 return result


# 生成 x 的值范围
x = np.linspace(0, 1, 400)

# 计算 yz 的值


z = q(x)
y = f(x)
if z[1]*y[1] > 0:
     y=y
else:
     y=-y

 #需要进行非线性变分    


# 创建图形
plt.plot(x, y, label='calculation')
plt.plot(x, z, label='real')


# 添加标题和标签
plt.title("Plot of calculation")
plt.xlabel("x")
plt.ylabel("y")

# 显示图例
plt.legend()

# 显示图形
plt.show()
