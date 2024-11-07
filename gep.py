import numpy as np
from scipy.linalg import qr
from scipy.linalg import eigh
import matplotlib.pyplot as plt
'''编写了简单的求解广义本征值问题的代码，会用到求解普通本征值问题的库函数，并在无限深势阱上进行测试'''

N=20
S=np.zeros((N,N))
H=np.zeros((N,N))

# 根据给定规则生成矩阵 H 和 S
for m in range(N):
    for n in range(N):
        if (m + n) % 2 == 0:  # m+n 为偶数的情况
            H[m, n] = -8 * (1-m-n-2*m*n) /((m+n+3)*(m+n+1)*(m+n-1))
            S[m, n] = 2 / (m + n + 5) - 4 / (m + n + 3) + 2 / (m + n + 1)
        else:  # m+n 为奇数的情况
            H[m, n] = 0
            S[m, n] = 0

print("矩阵 H:")
print(H)

print("矩阵 S:")
print(S)



  
eigenvalues, eigenvectors = eigh(S)

diagonal_matrix = np.diag(eigenvalues**(-0.5))

V = np.dot(eigenvectors , diagonal_matrix)
    
print("矩阵 eigenvalues:")
print(eigenvalues)

print("矩阵 eigenvectors:")
print(eigenvectors)

print("矩阵 diagonal_matrix:")
print(diagonal_matrix)

print("矩阵 V:")
print(V)


HH = np.dot(np.dot((V.T) , H ), (V))

print("矩阵 HH:")
print(HH)
eigenvalues, eigenvectors = eigh(HH)

print("eigenvalues:")
print(eigenvalues)

c1 = np.dot(V , eigenvectors[:,0])

# 求解的波函数与严格解之间的可视化
def f(x):
     result = 0 #初始化函数值为0
     for i in range(N):
         
         result += c1[i] * ((x**i)*(x-1)*(x+1))
        
     return result

j=1
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


