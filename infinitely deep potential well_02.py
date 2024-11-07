import numpy as np
from scipy.linalg import eigh  # 用于对称矩阵的广义特征值问题
import matplotlib.pyplot as plt
'''无限深势阱第一次修改后的代码，简单的尝试是，对于接近零的本征值，重新赋予一个很小的值，发现误差很大'''
# 定义矩阵的大小
N = 24 # 设定矩阵大小为 N x N
j = 5  # 定义第几个能级

# 初始化 A 和 B 矩阵
A = np.zeros((N, N))
B = np.zeros((N, N))
f = np.zeros((N))

# 根据给定规则生成矩阵 A 和 B
for m in range(N):
    for n in range(N):
        if (m + n) % 2 == 0:  # m+n 为偶数的情况
            A[m, n] = -8 * (1-m-n-2*m*n) /((m+n+3)*(m+n+1)*(m+n-1))
            B[m, n] = 2 / (m + n + 5) - 4 / (m + n + 3) + 2 / (m + n + 1)
        else:  # m+n 为奇数的情况
            A[m, n] = 0
            B[m, n] = 0



eigenvalues, eigenvectors = eigh(B)
eigenvalues[eigenvalues < 0] = 1e-10
     
diagonal_matrix = np.diag(eigenvalues**(-0.5)) ##这里会出大问题！！，随便设置一个本征值，然后计算，得到的误差会很大

V = np.dot(eigenvectors , diagonal_matrix)
    

H = np.dot(np.dot((V.T) , A ), (V))

print("矩阵 H:")
print(H)
eigenvalues, eigenvectors = eigh(H)

print("eigenvalues:")
print(eigenvalues)

c1 = np.dot(V , eigenvectors[:,j-1])

# 求解的波函数与严格解之间的可视化
def f(x):
     result = 0 #初始化函数值为0
     for i in range(N):
         
         result += c1[i] * ((x**i)*(x-1)*(x+1))
        
     return result



""" v_normalized = eigvecs / np.linalg.norm(eigvecs) """


        
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
