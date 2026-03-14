import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.font_manager as fm
import platform
import time

# ==========================================
# 0. 字体设置与辅助函数
# ==========================================
def configure_plotting_language():
    system = platform.system()
    if system == 'Windows':
        font_candidates = ['Microsoft YaHei', 'SimHei', 'SimSun']
    elif system == 'Darwin':
        font_candidates = ['Arial Unicode MS', 'Heiti TC', 'PingFang SC']
    else:
        font_candidates = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'SimHei', 'DejaVu Sans']
    
    available_fonts = set([f.name for f in fm.fontManager.ttflist])
    found_font = None
    for font in font_candidates:
        if font in available_fonts:
            found_font = font
            break
    
    plt.rcParams['axes.unicode_minus'] = False
    if found_font:
        plt.rcParams['font.sans-serif'] = [found_font] + plt.rcParams['font.sans-serif']
        return True
    else:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
        return False

USE_CHINESE = configure_plotting_language()

# ==========================================
# 1. 线性方程组求解算法集
# ==========================================

def solve_gaussian_elimination(A, b):
    """
    高斯消去法 (直接法)
    对应教材附录2.1
    """
    n = len(b)
    M = np.hstack([A, b.reshape(-1, 1)])  # 增广矩阵
    
    # 消元过程
    for i in range(n):
        # 选主元
        pivot = i + np.argmax(np.abs(M[i:, i]))
        if pivot != i:
            M[[i, pivot]] = M[[pivot, i]]
            
        # 消元
        for j in range(i + 1, n):
            factor = M[j, i] / M[i, i]
            M[j, i:] -= factor * M[i, i:]
            
    # 回代过程
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (M[i, -1] - np.dot(M[i, i+1:n], x[i+1:n])) / M[i, i]
        
    return x, []  # 直接法没有迭代历史，返回空列表

def solve_jacobi(A, b, tol=1e-8, max_iter=1000):
    """
    雅可比迭代法
    x^(k+1) = D^(-1) * (b - (L+U)*x^(k))
    """
    n = len(b)
    x = np.zeros(n)
    residuals = []
    
    D = np.diag(A)
    R = A - np.diag(D) # R = L + U
    
    for k in range(max_iter):
        # 计算残差 norm(b - Ax)
        r = b - np.dot(A, x)
        resid = np.linalg.norm(r)
        residuals.append(resid)
        
        if resid < tol:
            break
            
        # 迭代公式
        x = (b - np.dot(R, x)) / D
        
    return x, residuals

def solve_gauss_seidel(A, b, tol=1e-8, max_iter=1000):
    """
    高斯-赛德尔迭代法
    利用最新计算出的分量值
    """
    n = len(b)
    x = np.zeros(n)
    residuals = []
    
    for k in range(max_iter):
        r = b - np.dot(A, x)
        resid = np.linalg.norm(r)
        residuals.append(resid)
        
        if resid < tol:
            break
            
        x_new = np.copy(x)
        for i in range(n):
            # GS公式: x_i = (b_i - sum_{j<i} a_ij*x_j_new - sum_{j>i} a_ij*x_j_old) / a_ii
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]
        x = x_new
        
    return x, residuals

def solve_sor(A, b, omega=1.2, tol=1e-8, max_iter=1000):
    """
    逐次超松弛迭代法 (SOR)
    参考教材 2.5 节及 3.py
    """
    n = len(b)
    x = np.zeros(n)
    residuals = []
    
    for k in range(max_iter):
        r = b - np.dot(A, x)
        resid = np.linalg.norm(r)
        residuals.append(resid)
        
        if resid < tol:
            break
            
        for i in range(n):
            sigma = 0.0
            for j in range(n):
                if j != i:
                    sigma += A[i, j] * x[j]
            
            # Gauss-Seidel 预测值
            x_gs = (b[i] - sigma) / A[i, i]
            
            # 加权更新
            x[i] = (1 - omega) * x[i] + omega * x_gs
            
    return x, residuals

def solve_cg(A, b, tol=1e-8, max_iter=1000):
    """
    共轭梯度法 (CG)
    参考教材 2.6 节及 2.py
    要求 A 为对称正定矩阵
    """
    n = len(b)
    x = np.zeros(n)
    r = b - np.dot(A, x)
    p = r.copy()
    residuals = []
    
    for k in range(max_iter):
        r_norm = np.linalg.norm(r)
        residuals.append(r_norm)
        
        if r_norm < tol:
            break
            
        Ap = np.dot(A, p)
        alpha = np.dot(r, r) / np.dot(p, Ap)
        x = x + alpha * p
        r_new = r - alpha * Ap
        beta = np.dot(r_new, r_new) / np.dot(r, r)
        p = r_new + beta * p
        r = r_new
        
    return x, residuals

# ==========================================
# 2. 有限元主程序
# ==========================================
def fem_solver_compare():
    print("=" * 60)
    print("有限元法求解二维静电场 - 线性方程组求解算法对比")
    print("=" * 60)

    # --- 1. 定义几何模型与网格 (image_9f4687.png) ---
    # 节点坐标 (x, y)
    nodes = np.array([
        [2.0, 0.0], [5.0, 0.0], [2.0, 3.0], [5.0, 3.0], 
        [2.0, 5.0], [5.0, 5.0], [2.0, 7.0], [5.0, 7.0], 
        [0.0, 0.0], [0.0, 3.0], [0.0, 5.0], [0.0, 7.0], 
        [7.0, 0.0], [7.0, 3.0], [7.0, 5.0], [7.0, 7.0]
    ])
    
    # 单元连接 (0-based index)
    elements_raw = np.array([
        [15, 6, 4], [3, 4, 6], [1, 4, 3], [3, 5, 10], 
        [3, 6, 5], [5, 6, 8], [6, 16, 8], [6, 15, 16], 
        [4, 14, 15], [4, 13, 14], [4, 2, 13], [4, 1, 2], 
        [3, 9, 1], [3, 10, 9], [5, 11, 10], [5, 12, 11], 
        [5, 7, 12], [5, 8, 7]
    ])
    elements = elements_raw - 1
    num_nodes = len(nodes)
    
    # --- 2. 组装刚度矩阵 ---
    K_global = np.zeros((num_nodes, num_nodes))
    for elem in elements:
        idx_i, idx_j, idx_m = elem
        xi, yi = nodes[idx_i]; xj, yj = nodes[idx_j]; xm, ym = nodes[idx_m]
        
        # 计算 b, c 系数
        b_coef = np.array([yj - ym, ym - yi, yi - yj])
        c_coef = np.array([xm - xj, xi - xm, xj - xi])
        area = 0.5 * np.abs((xj - xi)*(ym - yi) - (xm - xi)*(yj - yi))
        
        # 单元刚度矩阵 Ke = (1/4A) * (bb^T + cc^T)
        Ke = np.zeros((3, 3))
        factor = 1.0 / (4.0 * area)
        for r in range(3):
            for c in range(3):
                Ke[r, c] = factor * (b_coef[r]*b_coef[c] + c_coef[r]*c_coef[c])
                K_global[elem[r], elem[c]] += Ke[r, c]

    # --- 3. 处理边界条件 ---
    # 节点 1-8 (Index 0-7) 是未知量 (待求)
    # 节点 9-16 (Index 8-15) 是已知边界
    free_nodes = np.arange(0, 8)
    fixed_nodes = np.arange(8, 16)
    
    # 边界值向量
    boundary_vals = np.zeros(num_nodes)
    boundary_vals[8:12] = 0.0  # 左边界
    boundary_vals[12:16] = 0.35 # 右边界
    
    # 提取子矩阵 K11, K12
    K11 = K_global[np.ix_(free_nodes, free_nodes)]
    K12 = K_global[np.ix_(free_nodes, fixed_nodes)]
    
    # 计算右端项 RHS = -K12 * U_fixed
    # (注意：源项 J=0，所以 R=0)
    RHS = -np.dot(K12, boundary_vals[fixed_nodes])
    
    print(f"待求解线性方程组规模: {K11.shape[0]} x {K11.shape[1]}")
    print("-" * 60)

    # --- 4. 多种方法求解与对比 ---
    solvers = {
        "Gaussian Elimination": (solve_gaussian_elimination, {}),
        "Jacobi Method":        (solve_jacobi, {'max_iter': 100}),
        "Gauss-Seidel Method":  (solve_gauss_seidel, {'max_iter': 100}),
        "SOR Method (w=1.2)":   (solve_sor, {'omega': 1.2, 'max_iter': 100}),
        "CG Method":            (solve_cg, {'max_iter': 100})
    }
    
    results = {}
    final_solution = None # 用于存储最终结果绘图
    
    # 设置绘图布局
    plt.figure(figsize=(16, 7))
    
    # 子图1: 残差收敛曲线
    ax1 = plt.subplot(1, 2, 1)
    
    for name, (solver_func, kwargs) in solvers.items():
        start_time = time.time()
        x, history = solver_func(K11, RHS, **kwargs)
        elapsed = (time.time() - start_time) * 1000 # ms
        
        results[name] = x
        final_solution = x # 暂存，用于后面画云图
        
        # 打印信息
        iter_count = len(history) if history else "N/A"
        final_resid = history[-1] if history else 0.0
        print(f"{name:<22} | 耗时: {elapsed:6.2f} ms | 迭代: {str(iter_count):<4} | 残差: {final_resid:.2e}")
        
        # 绘制收敛曲线 (仅迭代法)
        if history:
            ax1.semilogy(history, label=name, linewidth=1.5, marker='.', markersize=8)
    
    ax1.set_title("线性方程组求解算法收敛性对比" if USE_CHINESE else "Convergence Comparison of Linear Solvers")
    ax1.set_xlabel("迭代次数 (Iteration)" if USE_CHINESE else "Iteration")
    ax1.set_ylabel("残差范数 (Residual Norm)" if USE_CHINESE else "Residual Norm (log scale)")
    ax1.grid(True, which="both", linestyle='--', alpha=0.7)
    ax1.legend()

    # --- 5. 绘制最终物理场云图 ---
    # 合并解向量
    U_total = np.zeros(num_nodes)
    U_total[free_nodes] = results["Gaussian Elimination"] # 使用高斯消去法的解作为基准
    U_total[fixed_nodes] = boundary_vals[fixed_nodes]
    
    # 子图2: 电位分布
    ax2 = plt.subplot(1, 2, 2)
    ax2.set_aspect('equal')
    
    x = nodes[:, 0]
    y = nodes[:, 1]
    triang = tri.Triangulation(x, y, elements)
    
    # 绘制渐变云图 (levels设大一点以实现平滑渐变)
    contour = ax2.tricontourf(triang, U_total, levels=200, cmap='jet')
    # 【修改点】: 使用原始字符串 r'' 避免转义警告
    plt.colorbar(contour, ax=ax2, label=r'磁位 A (V·s/cm)')
    
    # 绘制网格线
    ax2.triplot(triang, 'w-', lw=0.5, alpha=0.5)
    
    # 标注节点
    # ax2.plot(x, y, 'ko', markersize=3)
    
    ax2.set_title("有限元计算结果: 磁位分布" if USE_CHINESE else "FEM Result: Potential Distribution")
    ax2.set_xlabel("X (cm)")
    ax2.set_ylabel("Y (cm)")

    plt.tight_layout()
    plt.show()

    # --- 6. 输出数值对比表 ---
    print("\n" + "="*60)
    print(f"{'节点':<6} {'高斯消去':<12} {'Jacobi':<12} {'GS':<12} {'SOR':<12} {'CG':<12}")
    print("-" * 70)
    for i in range(len(free_nodes)):
        node_id = free_nodes[i] + 1 # 1-based ID
        row_str = f"{node_id:<8d}"
        for name in solvers.keys():
            val = results[name][i]
            row_str += f"{val:<12.4f} "
        print(row_str)
    print("="*60)

if __name__ == "__main__":
    fem_solver_compare()