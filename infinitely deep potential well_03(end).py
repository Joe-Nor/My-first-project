import numpy as np
from scipy.linalg import eigh  # 用于对称矩阵的广义特征值问题
#from scipy.linalg import eig  # 如果矩阵 A 和 B 不对称，可以用这个
import matplotlib.pyplot as plt
'''无限深势阱的最终代码，删除接近平行的基组'''
# 定义矩阵的大小
N = 30 # 设定矩阵大小为 N x N
j = 5  # 定义第几个能级

H = np.zeros((N, N))
S = np.zeros((N, N))
f = np.zeros((N))

for m in range(N):
    for n in range(N):
        if (m + n) % 2 == 0:  # m+n 为偶数的情况
            H[m, n] = -8 * (1-m-n-2*m*n) /((m+n+3)*(m+n+1)*(m+n-1))
            S[m, n] = 2 / (m + n + 5) - 4 / (m + n + 3) + 2 / (m + n + 1)
        else:  # m+n 为奇数的情况
            H[m, n] = 0
            S[m, n] = 0

eigenvalues, eigenvectors = eigh(S)

K = np.sum(eigenvalues > 1e-15)
print("K:")
print(K)
V_V = np.zeros((N,K))

for i in range(K):
     V_V[:, i] = eigenvectors[:, N-K+i]/(np.sqrt(eigenvalues[N-K+i]))

H_H =  np.dot(np.dot( (V_V).conj().T , H ) , V_V )
S_S =  np.dot(np.dot( (V_V).conj().T , S ) , V_V )

'''print("H_H：")
print(H_H)

print("S_S：")
print(S_S)'''


eigvals, eigvecs = eigh(H_H, S_S)  
# 如果 A 和 B 不对称，使用 eig(H_H, S_S)
# 输出特征值和特征向量
'''print("特征值：")
print(eigvals)

print("特征向量：")
print(eigvecs)'''

""" v_normalized = eigvecs / np.linalg.norm(eigvecs) """
eigvecs= np.dot(V_V, eigvecs)
# 求解的波函数与严格解之间的可视化
def f(x):
     result = 0 #初始化函数值为0
     for i in range(N):
         
         result += eigvecs[i,j-1] * ((x**i)*(x-1)*(x+1))
        
     return result


def q(x): 
 result = 0 #初始化函数值为0
 if j % 2 == 0:  # j 为偶数的情况
            result = np.sin(np.pi * j *x /2)   
 else:  # j 为奇数的情况
           
         result = np.cos(np.pi * j * x /2)
       
 return result


# 生成 x 的值范围
x = np.linspace(-1, 1, 400)

# 计算 yz 的值

z = q(x)
y = f(x)
if z[1]*y[1] > 0:
     y=y
else:
     y=-y

     


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