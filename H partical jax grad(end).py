import jax
import jax.experimental
import jax.numpy as jnp  ## jax不支持 on-site 操作
from jax.scipy.linalg import eigh
import matplotlib.pyplot as plt
import optax
import numpy as np
'''通过自动微分，优化初始非线性参数'''
# 尝试使用jax.vmap来批处理，替代for循环
@jax.jit
def single_function(m, n, a):
    A_val = 3 * a[m] * a[n] * (jnp.pi)**1.5 / ((a[m]+a[n])**2.5) - 2 * jnp.pi/(a[m]+a[n])
    B_val = (jnp.pi / (a[m]+a[n]))**1.5
    return A_val, B_val

# jax无法解广义本征值问题，该函数输出最小本征值及归一化的本征矢
@jax.jit
def g_eigh_p(H, S):

     # 注释的这段是为了防止基函数过于接近，但是与jax.jit相性不和
     
     """eigenvalues, eigenvectors = eigh(S)
     K = jnp. sum(eigenvalues > 0) 
     V_V = jnp.zeros((H.shape[0],K))
     for i in range(K):
        V_V = V_V. at[:, i]. set(eigenvectors[:, H.shape[0]-K+i]/(jnp.sqrt(eigenvalues[H.shape[0]-K+i])))
     H_H =  jnp.dot(jnp.dot( (V_V).conj().T , H ) , V_V )
     S_S =  jnp.dot(jnp.dot( (V_V).conj().T , S ) , V_V )"""
     

     eigenvalues, eigenvectors = eigh(S)
     diagonal_matrix = jnp.diag(eigenvalues**(-0.5))
     V = jnp.matmul(eigenvectors, diagonal_matrix)
     H = jnp.matmul(jnp.matmul((V.T) , H ), V)
     eigenvalues, eigenvectors = eigh(H)
     """eigvecs = jnp.dot(V, eigenvectors)"""
     c = jnp.dot(V, eigenvectors)
     """if c.size == 0:
        raise ValueError("Computed eigenvector coefficients are empty.")"""
     return eigenvalues[0], c[:,0]
@jax.jit
def energy(alpha):
     A, B = jax.vmap(jax.vmap(single_function, in_axes=(0, None, None)), in_axes=(None, 0, None))(jnp.arange(N), jnp.arange(N), alpha)
     eigenvalues, _ = g_eigh_p(A, B)
     return eigenvalues

def energy2(alpha):
     A, B = jax.vmap(jax.vmap(single_function, in_axes=(0, None, None)), in_axes=(None, 0, None))(jnp.arange(N), jnp.arange(N), alpha)
     eigenvalues, eigenvectors = g_eigh_p(A, B)
     v_normalized = eigenvectors / jnp.sqrt(jnp.matmul(jnp.matmul(eigenvectors.T, B), eigenvectors))
     return eigenvalues, v_normalized

def grad(energy, alpha):
     value, vjp_fun = jax.vjp(energy, alpha)
     vjp = vjp_fun(1.0)[0]
     return value, vjp

N = 5

# 设置高斯分布的参数
mean = 0    # 均值
std_dev = 5 # 标准差
size = (N,)   # 生成的随机数数量
# 生成高斯随机数
# a = jnp.abs(np.random.normal(loc=mean, scale=std_dev, size=size))
a = jnp.array([0.10308952 , 0.32723364 , 34.06466  ,  1.1647596  ,  5.123983])
# a = jnp.array([13.00773 , 1.962079 , 0.444529 , 0.1219492])
# 设置裁剪值
clip_value = jnp.ones(N) # 设置最大和最小值
lr = 0.0001   # 学习率
num_steps = 100 # 迭代次数
optimizer = optax.adam(lr)
opt_state = optimizer.init(a)
for step in range(num_steps):
     loss_value, grads = grad(energy, a)  
     grads = jax.lax.clamp(grads, -clip_value, clip_value)
     print(loss_value, a)
     updates, opt_state = optimizer.update(grads, opt_state)
     a = optax.apply_updates(a, updates)
     # 确保更新后的 a 仍然是非负的
     a = jnp.abs(a)
 

print("基态能量：")
print("{:.20f}".format(energy(a))) # -0.49927866458892822266
print("非线性参数：")
print(a)

_ , v_normalized = energy2(a)

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