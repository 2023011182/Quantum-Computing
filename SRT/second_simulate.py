# 1. 导入必要的库
from qiskit_aer import Aer
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile
from qiskit.circuit.library import QFTGate, UnitaryGate
import numpy as np
from scipy.linalg import expm

# 2. 定义矩阵 A 和向量 b

N = 16
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
matrix_A = M_tilde
# 定义向量 b
vector_b = np.array([0.2625, 0.4375, 0.35, 0.175, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
normalized_b = vector_b / np.linalg.norm(vector_b)

# 3. 设置 HHL 参数
n_b = 4  
n_c = 6  # 保持高精度
n_a = 1  

# 计算演化时间 t
eigenvalues = np.linalg.eigvalsh(matrix_A)
lambda_max_abs = np.max(np.abs(eigenvalues))
scaling_factor = 0.9
t = (np.pi * scaling_factor) / lambda_max_abs

print(f"Time t: {t:.4f}")

# 4. 构建受控酉算子
def apply_controlled_u(qc, clock_qubits, input_qubits, matrix_A, t):
    for k in range(len(clock_qubits)):
        power = 2**k
        exponent = 1j * matrix_A * t * power
        u_matrix = expm(exponent)
        
        u_gate = UnitaryGate(u_matrix, label=f"U^{power}")
        c_u_gate = u_gate.control(1)
        qc.append(c_u_gate, [clock_qubits[k]] + input_qubits[:])

# 5. 构建 HHL 电路
clock = QuantumRegister(n_c, name='clock')
input_reg = QuantumRegister(n_b, name='b')
ancilla = QuantumRegister(n_a, name='ancilla')
measurement = ClassicalRegister(n_b, name='c')
meas_ancilla = ClassicalRegister(1, name='ma')

circuit = QuantumCircuit(ancilla, clock, input_reg, measurement, meas_ancilla)

# (A) 状态准备
circuit.initialize(normalized_b, input_reg)
circuit.barrier()

# (B) 量子相位估计 (QPE)
circuit.h(clock)
apply_controlled_u(circuit, clock, list(input_reg), matrix_A, t)

# (C) 逆量子傅里叶变换 (Inverse QFT)
iqft = QFTGate(num_qubits=n_c).inverse() 
circuit.append(iqft, clock)
circuit.barrier()

# (D) 辅助比特旋转 (Eigenvalue Inversion)
print("构建辅助比特旋转层...")

C = np.min(np.abs(eigenvalues))

for k in range(1, 2**n_c):
    if k < 2**(n_c - 1):
        k_signed = k 
    else:
        k_signed = k - 2**n_c 
    
    lambda_val = (2 * np.pi * k_signed) / (t * (2**n_c))
    
    if abs(lambda_val) < 1e-6: continue

    ratio = C / lambda_val
    if ratio > 1: ratio = 1
    if ratio < -1: ratio = -1
        
    theta = 2 * np.arcsin(ratio)
    
    k_binary = format(k, f'0{n_c}b')[::-1]
    
    for idx, bit in enumerate(k_binary):
        if bit == '0': circuit.x(clock[idx])
            
    circuit.mcry(theta, clock[:], ancilla[0])
    
    for idx, bit in enumerate(k_binary):
        if bit == '0': circuit.x(clock[idx])

circuit.barrier()

# (E) 逆量子相位估计 (Uncompute)
qft_gate = QFTGate(num_qubits=n_c)
circuit.append(qft_gate, clock)

# 逆酉演化
for k in reversed(range(len(clock))):
    power = 2**k
    exponent = -1j * matrix_A * t * power 
    u_matrix_dag = expm(exponent)
    
    u_gate_dag = UnitaryGate(u_matrix_dag, label=f"U_dag^{power}")
    c_u_gate_dag = u_gate_dag.control(1)
    circuit.append(c_u_gate_dag, [clock[k]] + list(input_reg))

circuit.h(clock)
circuit.barrier()

# (F) 测量
circuit.measure(ancilla, meas_ancilla)
circuit.measure(input_reg, measurement)

# 6. 运行模拟
print("开始运行模拟...")
simulator = Aer.get_backend('qasm_simulator')
transpiled_circuit = transpile(circuit, simulator)
job = simulator.run(transpiled_circuit, shots=50000)
result = job.result()
counts = result.get_counts()

# 7. 后处理
filtered_counts = {}
total_success_shots = 0

for key, val in counts.items():
    parts = key.split()
    if len(parts) == 2:
        ma_val, c_val = parts[0], parts[1]
    else:
        ma_val, c_val = key[0], key[1:]
        
    if ma_val == '1':
        filtered_counts[c_val] = val
        total_success_shots += val

print(f"成功次数 (Ancilla=1): {total_success_shots}")

# 8. 结果对比 (修改后：显示所有状态及相对误差)
print("\n--- 结果分析 (所有状态按顺序排列 0000 -> 1111) ---")
classic_x = np.linalg.solve(matrix_A, vector_b)
probs_classic = np.abs(classic_x / np.linalg.norm(classic_x))**2

if total_success_shots > 0:
    quantum_probs = {k: v / total_success_shots for k, v in filtered_counts.items()}
    
    # 打印表头，增加 Rel Diff 列
    print(f"{'State':<8} | {'Quantum Prob':<12} | {'Classical Prob':<14} | {'Diff':<10} | {'Rel Diff':<10}")
    print("-" * 70)
    
    # 循环遍历从 0 到 15 的所有索引
    for idx in range(2**n_b):
        bin_state = format(idx, f'0{n_b}b')
        q_prob = quantum_probs.get(bin_state, 0.0)
        c_prob = probs_classic[idx]
        
        diff = abs(q_prob - c_prob)
        
        # 计算相对误差： (Quantum - Classical) / Classical
        # 如果经典概率非常接近0，则显示 N/A 以避免除以零错误
        if c_prob > 1e-9:
            rel_diff = abs(q_prob - c_prob) / c_prob
            rel_diff_str = f"{rel_diff:.4f}" # 带符号显示
        else:
            rel_diff_str = "N/A"

        print(f"|{bin_state}> | {q_prob:.4f}       | {c_prob:.4f}         | {diff:.4f}     | {rel_diff_str}")
else:
    print("未测量到成功结果。")