import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

# ==========================================
# 0. 预处理：三对角化 (The Key Fix)
# ==========================================
def tridiagonalize(A):
    """
    使用 Householder 反射将对称矩阵 A 化为三对角矩阵 T。
    T 与 A 相似，拥有相同的特征值。
    """
    n = A.shape[0]
    A_curr = A.astype(np.float64).copy()
    
    for k in range(n - 2):
        # 选取第 k 列的主对角线下方元素
        x = A_curr[k+1:, k]
        norm_x = la.norm(x)
        if norm_x < 1e-15: continue # 已经是 0，跳过

        # 构造 Householder 向量 v
        sign = np.sign(x[0]) if x[0] != 0 else 1.0
        e1 = np.zeros_like(x)
        e1[0] = 1.0
        v = x + sign * norm_x * e1
        v = v / la.norm(v)

        # 构造反射矩阵 P (局部)
        P_sub = np.eye(len(x)) - 2 * np.outer(v, v)
        
        # 应用相似变换 P @ A @ P.T
        # 1. 左乘 (更新行)
        A_curr[k+1:, k:] = P_sub @ A_curr[k+1:, k:]
        # 2. 右乘 (更新列)
        A_curr[:, k+1:] = A_curr[:, k+1:] @ P_sub.T
        
    return A_curr

# ==========================================
# 1. QR 分解算法 (保持不变，用于迭代)
# ==========================================

def householder_qr(A):
    m, n = A.shape
    R = A.copy()
    Q = np.eye(m)
    for k in range(n - 1): # 对三对角矩阵，只需要处理 n-1 步
        x = R[k:m, k]
        norm_x = la.norm(x)
        if norm_x < 1e-15: continue
        sign = np.sign(x[0]) if x[0] != 0 else 1.0
        v = x + sign * norm_x * np.eye(len(x))[:, 0]
        v = v / la.norm(v)
        H_k = np.eye(len(x)) - 2 * np.outer(v, v)
        
        # 优化：只更新受影响的区域
        R[k:m, :] = H_k @ R[k:m, :]
        Q[:, k:m] = Q[:, k:m] @ H_k.T # 注意 Q 的更新方向
    return Q, R

def givens_qr(A):
    m, n = A.shape
    R = A.copy()
    Q = np.eye(m)
    # 对于三对角矩阵，Givens 只需要消除次对角线
    for j in range(n - 1):
        if abs(R[j+1, j]) < 1e-15: continue
        
        # 计算旋转参数
        a, b = R[j, j], R[j+1, j]
        if abs(b) > abs(a):
            r = a / b
            s = 1.0 / np.sqrt(1.0 + r**2)
            c = s * r
        else:
            r = b / a
            c = 1.0 / np.sqrt(1.0 + r**2)
            s = c * r
            
        # 构造旋转矩阵 G (局部 2x2)
        G = np.array([[c, -s], [s, c]])
        
        # 应用变换 (只更新第 j 和 j+1 行/列)
        R[j:j+2, :] = G.T @ R[j:j+2, :]
        Q[:, j:j+2] = Q[:, j:j+2] @ G
        
    return Q, R

def modified_gram_schmidt(A):
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    v = A.copy()
    
    for i in range(n):
        R[i, i] = la.norm(v[:, i])
        if R[i, i] < 1e-13:
            # 奇异性保护：如果模长太小，为了防止除以0，保持为0向量或跳过
            pass 
        else:
            Q[:, i] = v[:, i] / R[i, i]
            
        for j in range(i + 1, n):
            R[i, j] = np.dot(Q[:, i].T, v[:, j])
            v[:, j] = v[:, j] - R[i, j] * Q[:, i]
    return Q, R

# ==========================================
# 2. 矩阵构造与求解器
# ==========================================

def get_matrix_M():
    M = np.array([
        [-1/2, 25/12, 0, -10/12, 0, 0, 0, 0],
        [0, -10/12, -10/12, 50/12, 0, -15/12, 0, 0],
        [0, 0, 0, -15/12, -8/12, 50/12, 0, -15/12],
        [0, 0, 0, 0, 0, -15/12, -4/12, 25/12],
        [25/12, -1/2, -10/12, 0, 0, 0, 0, 0],
        [-10/12, 0, 50/12, -10/12, -15/12, 0, 0, 0],
        [0, 0, -15/12, 0, 50/12, -8/12, -15/12, 0],
        [0, 0, 0, 0, -15/12, 0, 25/12, -4/12]
    ])
    zero_8x8 = np.zeros((8, 8))
    M_tilde = np.block([
        [zero_8x8, M],
        [M.T, zero_8x8]
    ])
    return M_tilde

def find_eigenvalues_shifted(A, qr_func, max_iter=2000, tol=1e-6):
    n = A.shape[0]
    H = A.copy()
    eigvals = []
    
    for m in range(n, 0, -1):
        iter_count = 0
        while iter_count < max_iter:
            if m == 1: break
            # 收敛检查：检查次对角线元素
            if abs(H[m-1, m-2]) < tol: break
            
            # Wilkinson Shift (针对 2x2 块)
            d = H[m-1, m-1]
            b_elem = H[m-1, m-2]
            a = H[m-2, m-2]
            
            delta = (a - d) / 2.0
            sign = 1 if delta >= 0 else -1
            denom = abs(delta) + np.sqrt(delta**2 + b_elem**2)
            if denom == 0: denom = 1e-20
            mu = d - (sign * b_elem**2) / denom
            
            # QR 迭代步骤
            shift = mu * np.eye(m)
            try:
                # 尝试进行 QR 分解
                Q, R = qr_func(H[:m, :m] - shift)
                H[:m, :m] = R @ Q + shift
            except:
                break # 如果 MGS 失败，中断
            iter_count += 1
            
        eigvals.append(H[m-1, m-1])
        
    return np.array(eigvals)

# ==========================================
# 3. 主程序
# ==========================================

def main():
    M = get_matrix_M()
    print(f"Matrix M shape: {M.shape}")
    
    # 1. Ground Truth
    eig_numpy = np.sort(la.eigvals(M))
    
    # 2. 先进行三对角化 (Fix 核心步骤)
    print("Step 1: Reducing to Tridiagonal Form...")
    T = tridiagonalize(M)
    
    # 3. 在三对角矩阵 T 上运行 QR 迭代
    print("Step 2: Running QR Algorithms on Tridiagonal Matrix...")
    
    print("  -> Householder QR...")
    eig_hh = np.sort(find_eigenvalues_shifted(T, householder_qr))
    
    print("  -> Givens QR...")
    eig_givens = np.sort(find_eigenvalues_shifted(T, givens_qr))
    
    print("  -> Modified GS QR...")
    # MGS 在三对角矩阵上比在稠密矩阵上更稳定
    eig_mgs = np.sort(find_eigenvalues_shifted(T, modified_gram_schmidt))

    # 打印结果表
    print("\n{:<10} {:<15} {:<15} {:<15} {:<15}".format("Index", "Numpy (Ref)", "Householder", "Givens", "MGS"))
    print("-" * 75)
    for i in range(len(eig_numpy)):
        # 处理可能因算法失败产生的 0 填充 (MGS 如果仍有不稳定)
        val_hh = eig_hh[i] if i < len(eig_hh) else 0
        val_giv = eig_givens[i] if i < len(eig_givens) else 0
        val_mgs = eig_mgs[i] if i < len(eig_mgs) else 0
        
        print("{:<10d} {:<15.6f} {:<15.6f} {:<15.6f} {:<15.6f}".format(
            i+1, eig_numpy[i], val_hh, val_giv, val_mgs
        ))

    # 绘图
    plt.figure(figsize=(10, 6))
    x_axis = np.arange(1, 17)
    plt.plot(x_axis, eig_numpy, 'k-', linewidth=3, alpha=0.3, label='Numpy (Reference)')
    plt.scatter(x_axis, eig_hh, c='red', marker='x', s=80, label='Householder (Fixed)')
    plt.scatter(x_axis, eig_givens, c='blue', marker='+', s=80, label='Givens (Fixed)')
    plt.scatter(x_axis, eig_mgs, edgecolors='green', facecolors='none', marker='o', s=50, label='MGS (Fixed)')
    
    plt.title('Eigenvalues Comparison (After Tridiagonalization Fix)')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()