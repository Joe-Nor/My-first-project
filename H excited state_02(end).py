import jax
import jax.experimental
import jax.numpy as jnp  ## jax不支持 on-site 操作
from jax.scipy.linalg import eigh
import matplotlib.pyplot as plt
import optax
import numpy as np
'''通过拉格朗日乘子法，加上一条等式约束，即基态波函数与激发态波函数要正交。为了减少手推的繁琐，基态波函数选为之前训练好的结果，这样只相差一个交叠矩阵'''
@jax.jit
def single_function(m, n, a):
    A_val = 3 * a[m] * a[n] * (jnp.pi)**1.5 / ((a[m]+a[n])**2.5) - 2 * jnp.pi/(a[m]+a[n])
    B_val = (jnp.pi / (a[m]+a[n]))**1.5
    return A_val, B_val
@jax.jit
def single_function_2(m, n, b):
    C_val = (jnp.pi / (alpha[m]+b[n]))**1.5
    return C_val
@jax.jit
def g_eigh_p(H, S):
     eigenvalues, eigenvectors = eigh(S)
     diagonal_matrix = jnp.diag(eigenvalues**(-0.5))
     V = jnp.matmul(eigenvectors, diagonal_matrix)
     H = jnp.matmul(jnp.matmul((V.T) , H ), V)
     eigenvalues, eigenvectors = eigh(H)
     c = jnp.dot(V, eigenvectors)
     return eigenvalues[0], c[:,0]
@jax.jit
def g_eigh_p_2(H, S):
     eigenvalues, eigenvectors = eigh(S)
     diagonal_matrix = jnp.diag(eigenvalues**(-0.5))
     V = jnp.matmul(eigenvectors, diagonal_matrix)
     H = jnp.matmul(jnp.matmul((V.T) , H ), V)
     eigenvalues, eigenvectors = eigh(H)
     c = jnp.dot(V, eigenvectors)
     return eigenvalues, c
@jax.jit
def energy(alpha):
     A, B = jax.vmap(jax.vmap(single_function, in_axes=(0, None, None)), in_axes=(None, 0, None))(jnp.arange(N), jnp.arange(N), alpha)
     eigenvalues, _ = g_eigh_p(A, B)
     return eigenvalues
@jax.jit
def energy2(alpha):
     A, B = jax.vmap(jax.vmap(single_function, in_axes=(0, None, None)), in_axes=(None, 0, None))(jnp.arange(N), jnp.arange(N), alpha)
     eigenvalues, eigenvectors = g_eigh_p(A, B)
     v_normalized = eigenvectors / jnp.sqrt(jnp.matmul(jnp.matmul(eigenvectors.T, B), eigenvectors))
     return eigenvalues, v_normalized
@jax.jit
def energy3(beta, l):
     A, B = jax.vmap(jax.vmap(single_function, in_axes=(0, None, None)), in_axes=(None, 0, None))(jnp.arange(M), jnp.arange(M), beta)
     C = jax.vmap(jax.vmap(single_function_2, in_axes=(None, 0, None)), in_axes=(0, None, None))(jnp.arange(M), jnp.arange(N), beta)
     eigenvalues, eigenvectors = g_eigh_p_2(A, B)
     v_normalized = eigenvectors[:,1] / jnp.sqrt(jnp.matmul(jnp.matmul(eigenvectors[:,1].T, B), eigenvectors[:,1]))
     F = eigenvalues[1] + l * jnp.matmul(jnp.matmul(v_normalized.T, C), v_a)
     return F
@jax.jit
def energy4(beta):
     A, B = jax.vmap(jax.vmap(single_function, in_axes=(0, None, None)), in_axes=(None, 0, None))(jnp.arange(M), jnp.arange(M), beta)
     eigenvalues, eigenvectors = g_eigh_p_2(A, B)
     v_normalized1 = eigenvectors[:,0] / jnp.sqrt(jnp.matmul(jnp.matmul(eigenvectors[:,0].T, B), eigenvectors[:,0]))
     v_normalized2 = eigenvectors[:,1] / jnp.sqrt(jnp.matmul(jnp.matmul(eigenvectors[:,1].T, B), eigenvectors[:,1]))
     return eigenvalues[0], eigenvalues[1], v_normalized1, v_normalized2
def grad(energy, alpha):
     value, vjp_fun = jax.vjp(energy, alpha)
     vjp = vjp_fun(1.0)[0]
     return value, vjp



N = 5 # 算基态的基函数数目
### 变分算基态能的优化后参数alpha
a = jnp.array([0.10308952 , 0.32723364 , 34.06466  ,  1.1647596  ,  5.123983])
### 训练基态的参数
# 定义下降函数，并保证参数非负
clip_value = jnp.ones(N) # 设置最大和最小值
lr = 0.0001   # 学习率
num_steps = 10 # 迭代次数
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
alpha = a
E_a , v_a = energy2(alpha)
"""print("基态能量：")
print("{:.20f}".format(E_a)) # -0.49927866458892822266
print("非线性参数alpha：")
print(alpha)
print("v_a：")
print(v_a)"""


def optimize_with_lagrange(params_init, l_init, learning_rate_1, learning_rate_2, num_steps):
    # 初始化 Optax 优化器
    optimizer_params = optax.adam(learning_rate_1)
    optimizer_lambda = optax.adam(learning_rate_2)
    opt_state_params = optimizer_params.init(params_init)
    opt_state_lambda = optimizer_lambda.init(l_init)

    params, l = params_init, l_init

    for step in range(num_steps):
        # 计算拉格朗日函数相对于 params 和 lambda 的 vjp
        lagrangian_grad_fn = lambda params, l: energy3(params, l)
        value, vjp_fn_params = jax.vjp(lagrangian_grad_fn, params, l)
        print(f"Step {step}, Lagrangian Value: {value}, params: {params}")
        # 计算相对于参数和 lambda 的梯度
        grad_params, grad_l = vjp_fn_params(jnp.ones_like(energy3(params, l)))
        grad_params = jax.lax.clamp(grad_params, -clip_value, clip_value)
        grad_l = jax.lax.clamp(grad_l, -clip_value_2, clip_value_2)
        # 使用 Optax 优化器更新参数
        updates_params, opt_state_params = optimizer_params.update(grad_params, opt_state_params)
        params = optax.apply_updates(params, updates_params)
        params = jnp.abs(params)
        # 手动更新 lambda（拉格朗日乘子，用梯度上升来满足约束）
        grad_l = -grad_l
        updates_lambda, opt_state_lambda = optimizer_lambda.update(grad_l, opt_state_lambda)
        l = optax.apply_updates(l, updates_lambda)
        # l = jnp.abs(l)
    return params, l


M = 6 # 算第一激发态的基函数数目
# 设置高斯分布的参数
mean = 0    # 均值
std_dev = 10 # 标准差
size = (M,)   # 生成的随机数数量M
### 激发态的参数beta
# b_init = jnp.abs(np.random.normal(loc=mean, scale=std_dev, size = size))
# b_init = jnp.array([2.0264899e-02, 2.0958337e-01, 3.3787701e+01, 8.0792391e-01, 4.1075797e+00])
b_init = jnp.array([2.0126490e-02, 2.2147416e-01, 9.2454219e-01, 4.9110332e+00, 4.2821609e+01, 1.0072286e+03])
print("非线性参数beta：")
print(b_init)
### 初始化拉格朗日乘子
l_init = 0.0
b, l = b_init, l_init
### 训练激发态的参数
# 定义下降函数，并保证参数非负
clip_value = jnp.ones(M) # 设置最大和最小值
clip_value_2 = jnp.ones(1)
lr_1 = 0.0000001   # 学习率
lr_2 = 0.000000001
num_steps = 10000 # 迭代次数
b_opt, l_opt = optimize_with_lagrange(b_init, l_init, lr_1, lr_2, num_steps)
E_a, E_b , v_a, v_b = energy4(b_opt)
print("基态能量：")
print("{:.20f}".format(E_a)) 
print("激发态能量：")
print("{:.20f}".format(E_b)) 
print("非线性参数beta：")
print(b_opt)
print(l_opt)





@jax. jit
def f1(x):
     result = 0 #初始化函数值为0
     for i in range(N):
         result += v_a[i] * jnp.exp(- a[i] * (x**2) )
        
     return result


@jax. jit
def q1(x): 
 result = 2 * jnp.exp(-x)  /jnp.sqrt(4*jnp.pi)
 return result


# 生成 x 的值范围
x = jnp.linspace(0, 10, 4000)

# 计算 yz 的值


z1 = q1(x)
y1 = f1(x)
if z1[1]*y1[1] > 0:
     y1=y1
else:
     y1=-y1
  


@jax. jit
def f2(x):
     result = 0 #初始化函数值为0
     for i in range(M):
         result += v_b[i] * jnp.exp(- b_opt[i] * (x**2) )
        
     return result


@jax. jit
def q2(x): 
 result = 1/jnp.sqrt(2) * (1-x/2) * jnp.exp(-x/2)  /jnp.sqrt(4*jnp.pi)
 return result



# 计算 yz 的值


z2 = q2(x)
y2 = f2(x)
if z2[1]*y2[1] > 0:
     y2=y2
else:
     y2=-y2
  


# 创建图形
plt.plot(x, y1, label='calculation_a')
plt.plot(x, z1, label='real_a')
plt.plot(x, y2, label='calculation_b')
plt.plot(x, z2, label='real_b')

# 添加标题和标签
plt.title("Plot of calculation")
plt.xlabel("x")
plt.ylabel("y")

# 显示图例
plt.legend()

# 显示图形
plt.show()