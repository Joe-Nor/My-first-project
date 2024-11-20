import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from jax import jit

L = 10
N = 10
R = 1
r = R / (2 * jnp.sqrt(3))

# 网格点坐标
a = jnp.arange(1, 2 * N + 2)
x_a = L / (2 * N + 1) * (a - N - 1)
y_a = x_a
z_a = x_a

# 计算 t 矩阵 (动能部分)
m, n = jnp.meshgrid(jnp.arange(2 * N + 1), jnp.arange(2 * N + 1), indexing="ij")
t_off_diag = -(2 * jnp.pi / L) ** 2 * (-1) ** (m - n) * jnp.cos(jnp.pi * (m - n) / (2 * N + 1)) / (4 * (jnp.sin(jnp.pi * (m - n) / (2 * N + 1))) ** 2)
t_diag = -(2 * jnp.pi / L) ** 2 * N * (N + 1) / 6
t = jnp.where(m == n, t_diag, t_off_diag)

# 构造三维空间动能矩阵 T
I = jnp.eye(2 * N + 1)
T = jnp.kron(jnp.kron(t, I), I) + jnp.kron(jnp.kron(I, t), I) + jnp.kron(jnp.kron(I, I), t)

# 单体相互作用势能 V_diag
coords = jnp.stack(jnp.meshgrid(x_a, y_a, z_a, indexing="ij"), axis=-1).reshape(-1, 3)
dist1 = jnp.linalg.norm(coords - jnp.array([r, r, r]), axis=1)
dist2 = jnp.linalg.norm(coords + jnp.array([r, r, r]), axis=1)
V_diag = -1 / dist1 - 1 / dist2
V = jnp.diag(V_diag)
H_core = T + V
# 构造稀疏距离矩阵 D
@jit
def compute_distance_matrix(coords):
    diff = coords[:, None, :] - coords[None, :, :]
    distances = jnp.linalg.norm(diff, axis=-1)
    distances = jnp.where(distances == 0, jnp.inf, distances)  # 避免除以零
    distances = 1 / distances
    return distances

D_dense = compute_distance_matrix(coords)

# 计算二体相互作用 G_diag
@jit
def compute_G_diag(C, D_dense):
    # 计算 C 的共轭转置
    C_conj = jnp.conj(C)
    # 执行批量矩阵乘法来计算 G_diag
    # 计算 G_diag = C^T * D_dense * C，其中 C^T 是 C 的共轭转置
    G_diag = jnp.sum(C_conj[:, None] * D_dense * C[None, :], axis=0)
    return G_diag


def scf_iteration(H_core, D_dense, max_iter=1000, tol=1e-6):
    """
    Args:
        H_core: 单电子哈密顿矩阵 (n_basis, n_basis)
        max_iter: 最大迭代次数
        tol: 收敛容忍度
    Returns:
        最终的Fock矩阵, 线性系数, SCF能量
    """
    n_basis = H_core.shape[0]
    C = jnp.ones((n_basis,))  # 假设初始系数全为 0

    for iteration in range(max_iter):
        
        density_matrix = jnp.outer(jnp.conj(C) , C)
        G = jnp.diag(compute_G_diag(C, D_dense))
        _ , C_prime = jnp.linalg.eigh(H_core + G)
        new_density_matrix = jnp.outer(jnp.conj(C_prime[:,0]), (C_prime[:,0]))
        C = C_prime[:,0]
        # 计算SCF能量
        E_elec = jnp.sum(density_matrix * (2 * H_core + G) ) # SCF电子能量

        # 检查收敛性
        delta = jnp.linalg.norm(new_density_matrix - density_matrix)
        print("第", iteration, "次循环:", float(delta))
        if float(delta) < tol:  # 强制转换为具体值
            print(f"SCF收敛于第 {iteration + 1} 次迭代，能量: {E_elec:.12f} Hartree")
            return new_density_matrix, E_elec

    raise RuntimeError("SCF未能收敛")

# 调用SCF迭代
P_final, E_final = scf_iteration(H_core, D_dense)
print("最终SCF总能量:", E_final)


'''from jax import jit
import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO



L = 10
N = 1
R = 1
r = R / (2*jnp.sqrt(3))  # 对称起见，AB两原子均位于坐标系对角线上
a = jnp.arange(1, 2 * N + 2)  # 包含 1 到 2N+1 的整数
# 计算 x(a) = L / (2N+1) * (a - N - 1)
x_a = L / (2 * N + 1) * (a - N - 1)
y_a = L / (2 * N + 1) * (a - N - 1)
z_a = L / (2 * N + 1) * (a - N - 1)

t = jnp.zeros((2*N+1, 2*N+1))
I = jnp.eye(2*N+1)
for m in range(2*N+1):
    for n in range(2*N+1):
        if m != n:  # 非对角元
           t = t.at[m,n].set( - (2*jnp.pi/L)**2 * (-1)**(m-n) * jnp.cos(jnp.pi * (m-n) / (2*N+1)) / ( 4*(jnp.sin(jnp.pi * (m-n) / (2*N+1)))**2 ) ) 
        else:  # 对角元
           t = t.at[m,n].set( - (2*jnp.pi/L)**2 * N * (N+1) /6 ) 
print(t)

T = jnp.kron( jnp.kron(t , I) , I) + jnp.kron( jnp.kron(I , t) , I) + jnp.kron( jnp.kron(I , I) , t)  # 单体相互作用动能部分

V_diag = jnp.zeros(((2*N+1)**3))
for i in range(2*N+1):
    for j in range(2*N+1):
        for k in range(2*N+1):
           V_diag = V_diag.at[i*(2*N+1)*(2*N+1)+j*(2*N+1)+k].set( - 1 / jnp.sqrt((x_a[i]-r)**2 + (y_a[j]-r)**2 + (z_a[k]-r)**2) - 1 / jnp.sqrt((x_a[i]+r)**2 + (y_a[j]+r)**2 + (z_a[k]+r)**2) )

V = jnp.diag(V_diag) # 单体相互作用势能部分
C = jnp.zeros(((2*N+1)**3)) # 初始系数，设为0

D= jnp.zeros((2*N+1,2*N+1,2*N+1,2*N+1,2*N+1,2*N+1)) # 1/d距离矩阵
for a in range(2*N+1):
    for b in range(2*N+1):
        for c in range(2*N+1):
            for i in range(2*N+1):
                for j in range(2*N+1):
                   for k in range(2*N+1):
                        if a!=i or b!=j or c!=k:
                            D = D.at[a,b,c,i,j,k].set(1/jnp.sqrt((x_a[i]-x_a[a])**2 + (y_a[j]-y_a[b])**2 + (z_a[k]-z_a[c])**2))

G_diag = jnp.zeros(((2*N+1)**3)) 
d = jnp.zeros(((2*N+1)**3))
@jit
def get_G(G_diag,d,C):
    for a in range(2*N+1):
        for b in range(2*N+1):
             for c in range(2*N+1):
                  for i in range(2*N+1):
                       for j in range(2*N+1):
                             for k in range(2*N+1):
                                  d = d.at[i*(2*N+1)*(2*N+1)+j*(2*N+1)+k].set( D[a,b,c,i,j,k] )
                                  G_diag = G_diag.at[a*(2*N+1)*(2*N+1)+b*(2*N+1)+c].set(jnp.matmul(jnp.matmul((jnp.conj(C).T) , jnp.diag(d) ), C))
    return G_diag

G = jnp.diag(get_G(G_diag,d,C)) # 二体相互作用势能项

print(G.shape)
@jit
def compute_fock_matrix(H_core, density_matrix, two_electron_integrals):
    """
    计算Fock矩阵
    Args:
        H_core: 单电子哈密顿矩阵 (n_basis, n_basis)
        density_matrix: 当前密度矩阵 (n_basis, n_basis)
        two_electron_integrals: 四中心积分 (n_basis, n_basis, n_basis, n_basis)
    Returns:
        Fock矩阵 (n_basis, n_basis)
    """
    # Hartree项
    hartree = jnp.einsum('kl,ijkl->ij', density_matrix, two_electron_integrals)
    # 交换项
    exchange = -0.5 * jnp.einsum('kl,ilkj->ij', density_matrix, two_electron_integrals)
    return H_core + hartree + exchange'''




