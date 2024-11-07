import jax
import jax.numpy as jnp  ## jax不支持 on-site 操作
from jax.scipy.linalg import eigh
import matplotlib.pyplot as plt
'''氢原子基态能求解的jax版本，由于jax没有求解广义本征值问题的库函数，需要用到gep.py中的部分代码'''
a = jnp.array([13.00773 , 1.962079 , 0.444529 , 0.1219492])
N = len(a)
f = jnp.zeros((N))

#尝试使用jax.vmap来批处理，替代for循环
@jax.jit
def single_function(m,n):
    A_val = 3 * a[m] * a[n] * (jnp.pi)**1.5 / ((a[m]+a[n])**2.5) - 2 * jnp.pi/(a[m]+a[n])
    B_val = (jnp.pi / (a[m]+a[n]))**1.5
    return A_val, B_val

    
A, B = jax.vmap(jax.vmap(single_function, in_axes=(0, None)), in_axes=(None, 0))(jnp.arange(N), jnp.arange(N))

#来自gep.py
eigenvalues, eigenvectors = eigh(B)
diagonal_matrix = jnp.diag(eigenvalues**(-0.5))
V = jnp.matmul(eigenvectors , diagonal_matrix)   
H = jnp.matmul(jnp.matmul((V.T) , A ), V)
eigenvalues, eigenvectors = eigh(H)


#基态对应的特征向量
c1 = jnp.matmul(V , eigenvectors[:,0])


print("特征值：")
print("{:.20f}".format(eigenvalues[0]))


v_normalized = c1 / jnp.sqrt(jnp.matmul(jnp.matmul(c1.T,B), c1))
@jax. jit
def f(x):
     result = 0 #初始化函数值为0
     for i in range(N):
         result += v_normalized[i] * jnp.exp(- a[i] * (x**2) )
        
     return result


@jax. jit
def q(x): 
 result = 2 * jnp.exp(-x)  /jnp.sqrt(4*jnp.pi)
 return result


# 生成 x 的值范围
x = jnp.linspace(0, 1, 400)

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
