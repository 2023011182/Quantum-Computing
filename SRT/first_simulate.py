import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

# Qiskit 基础库
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile
from qiskit.circuit.library import QFTGate, UnitaryGate

# Qiskit Aer (本地模拟器)
from qiskit_aer import AerSimulator
# 结果可视化（可选）
from qiskit.visualization import plot_histogram

# ==========================================
# 1. 定义矩阵 A 和向量 b (8x8系统)
# ==========================================
print(">>> 1. 初始化数学参数...")

# 定义 8x8 矩阵 (来源于图片数据)
matrix_A = np.array([
    [25/12, -1/2, -10/12, 0, 0, 0, 0, 0],
    [-1/2, 25/12, 0, -10/12, 0, 0, 0, 0],
    [-10/12, 0, 50/12, -10/12, -15/12, 0, 0, 0],
    [0, -10/12, -10/12, 50/12, 0, -15/12, 0, 0],
    [0, 0, -15/12, 0, 50/12, -8/12, -15/12, 0],
    [0, 0, 0, -15/12, -8/12, 50/12, 0, -15/12],
    [0, 0, 0, 0, -15/12, 0, 25/12, -4/12],
    [0, 0, 0, 0, 0, -15/12, -4/12, 25/12]
])

# 定义向量 b
vector_b = np.array([0, 0.2625, 0, 0.4375, 0, 0.35, 0, 0.175])

# 归一化向量 b
norm_b = np.linalg.norm(vector_b)
normalized_b = vector_b / norm_b

# 验证厄米性
is_hermitian = np.allclose(matrix_A, matrix_A.conj().T)
if not is_hermitian:
    print("    [警告] 矩阵不是厄米的，模拟结果可能不准确。")

# ==========================================
# 2. 设置 HHL 参数
# ==========================================
N = len(vector_b)
n_b = int(np.log2(N))  # 3 qubits
n_c = 4                # 时钟比特 (可增加到 5-6 以提高模拟精度)
n_a = 1                # 辅助比特

# 计算特征值范围
eigenvalues = np.linalg.eigvalsh(matrix_A)
lambda_min = np.min(np.abs(eigenvalues))
lambda_max = np.max(np.abs(eigenvalues))

# 设定演化时间 t 和旋转常数 C
# 模拟器没有退相干限制，我们可以追求更高的理论精度
t = (2 * np.pi * 0.95) / lambda_max 
C_val = lambda_min * 0.99

print(f"    qubits: n_b={n_b}, n_c={n_c}")
print(f"    t={t:.4f}, C={C_val:.4f}")

# ==========================================
# 3. 构建 HHL 电路
# ==========================================
print(">>> 2. 构建量子电路...")

clock = QuantumRegister(n_c, name='clock')
input_reg = QuantumRegister(n_b, name='b')
ancilla = QuantumRegister(n_a, name='ancilla')
# 只需要测量 b 和 ancilla
measurement = ClassicalRegister(n_b + 1, name='c') 

circuit = QuantumCircuit(ancilla, clock, input_reg, measurement)

# (A) 状态制备
circuit.initialize(normalized_b, input_reg)
circuit.barrier()

# (B) QPE
circuit.h(clock)
for k in range(n_c):
    power = 2**k
    exponent = 1j * matrix_A * t * power
    u_matrix = expm(exponent)
    u_gate = UnitaryGate(u_matrix, label=f"U^{power}")
    circuit.append(u_gate.control(1), [clock[k]] + list(input_reg))

circuit.append(QFTGate(num_qubits=n_c).inverse(), clock)
circuit.barrier()

# (C) 特征值反演 (Rotation)
for k in range(1, 2**n_c):
    lam = k * (2 * np.pi) / (t * (2**n_c))
    if abs(lam) < 1e-6: continue
    
    ratio = C_val / lam
    ratio = max(-1.0, min(1.0, ratio))
    theta = 2 * np.arcsin(ratio)
    
    k_binary = format(k, f'0{n_c}b')[::-1]
    
    # 多控旋转
    for idx, bit in enumerate(k_binary):
        if bit == '0': circuit.x(clock[idx])
    circuit.mcry(theta, clock[:], ancilla[0])
    for idx, bit in enumerate(k_binary):
        if bit == '0': circuit.x(clock[idx])

circuit.barrier()

# (D) 逆 QPE
circuit.append(QFTGate(num_qubits=n_c), clock)
for k in reversed(range(n_c)):
    power = 2**k
    exponent = -1j * matrix_A * t * power 
    u_matrix_dag = expm(exponent)
    u_gate_dag = UnitaryGate(u_matrix_dag, label=f"U_dag^{power}")
    circuit.append(u_gate_dag.control(1), [clock[k]] + list(input_reg))
circuit.h(clock)
circuit.barrier()

# (E) 测量
# 测量 Ancilla (存入 c[0]) 和 b (存入 c[1]...c[n_b])
circuit.measure(ancilla, measurement[0])
circuit.measure(input_reg, measurement[1:])

# ==========================================
# 4. 在本地模拟器上运行
# ==========================================
print(">>> 3. 在本地 AerSimulator 上运行...")

# 初始化模拟器
backend = AerSimulator()

# 转译电路 (将 UnitaryGate 分解为基础门)
# 模拟器不需要过于激进的优化，basis_gates 会自动适配
isa_circuit = transpile(circuit, backend)

# 执行任务
job = backend.run(isa_circuit, shots=10000)
result = job.result()
counts = result.get_counts()

# ==========================================
# 5. 结果分析
# ==========================================
print(">>> 4. 结果分析...")

# 过滤 Ancilla = 1 的结果
# 测量结果 key 格式为 "b_bits ancilla_bit" 或 "b3b2b1 a"
# 因为 measurement[0] 是 ancilla，它在 bitstring 的最右边
count_success = 0
filtered_counts = {}

for key, val in counts.items():
    # 假设 key 是类似 "101 1" 或 "1011"
    # 我们取最右边一位作为 ancilla
    ancilla_bit = key[-1]
    b_state = key[:-1].strip() # 去除可能的空格
    
    if ancilla_bit == '1':
        count_success += val
        filtered_counts[b_state] = filtered_counts.get(b_state, 0) + val

print(f"\n后选择成功次数 (Ancilla=1): {count_success} / 10000")

print("\n[模拟器解分布] (按概率排序):")
# 转换为概率并排序
sorted_results = sorted(filtered_counts.items(), key=lambda x: x[1], reverse=True)
for state, count in sorted_results:
    prob = count / count_success
    print(f"  |{state}> : {prob:.4f}")

# ==========================================
# 6. 与理论经典解对比
# ==========================================
print("\n[经典理论解]:")
x_exact = np.linalg.solve(matrix_A, vector_b)
probs_exact = np.abs(x_exact / np.linalg.norm(x_exact))**2

# 为了方便对比，将理论解转换为二进制字符串索引
exact_dict = {}
for i, p in enumerate(probs_exact):
    bin_str = format(i, f'0{n_b}b')
    exact_dict[bin_str] = p
    
# 打印对比 (只打印理论概率 > 0.001 的项)
for state, prob in exact_dict.items():
    if prob > 0.001:
        print(f"  |{state}> : {prob:.4f}")

# 可视化对比 (如果是在 Jupyter Notebook 中)
# plot_histogram([filtered_counts])
# plt.show()