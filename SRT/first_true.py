import numpy as np
from scipy.linalg import expm

# Qiskit 库
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit.library import QFTGate, UnitaryGate
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# ==========================================
# 1. 定义矩阵 A 和向量 b
# ==========================================
print(">>> 1. 初始化数学参数...")

# 定义 8x8 矩阵
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

# ==========================================
# 2. 设置 HHL 参数
# ==========================================
N = len(vector_b)
n_b = int(np.log2(N))  # 3 qubits
n_c = 4                # 时钟比特
n_a = 1                # 辅助比特

# 计算特征值
eigenvalues = np.linalg.eigvalsh(matrix_A)
lambda_min_abs = np.min(np.abs(eigenvalues))
lambda_max_abs = np.max(np.abs(eigenvalues))

t = (2 * np.pi * 0.9) / lambda_max_abs
C_val = lambda_min_abs * 0.99 

print(f"    t: {t:.4f}, C: {C_val:.4f}")

# ==========================================
# 3. 构建 HHL 电路
# ==========================================
print(">>> 2. 构建量子电路...")

clock = QuantumRegister(n_c, name='clock')
input_reg = QuantumRegister(n_b, name='b')
ancilla = QuantumRegister(n_a, name='ancilla')
# 修改：将所有测量结果放入同一个经典寄存器，方便关联筛选
# 顺序：低位是 ancilla，高位是 input_reg (b)
measurement = ClassicalRegister(n_b + 1, name='meas') 

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

# (C) 受控旋转
for k in range(1, 2**n_c):
    lam = k * (2 * np.pi) / (t * (2**n_c))
    if abs(lam) < 1e-6: continue
    ratio = max(-1.0, min(1.0, C_val / lam))
    theta = 2 * np.arcsin(ratio)
    k_binary = format(k, f'0{n_c}b')[::-1]
    
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
# 关键修改：明确测量顺序
# meas[0] <- ancilla (作为标志位)
# meas[1...n_b] <- input_reg (作为解向量)
circuit.measure(ancilla, measurement[0])
circuit.measure(input_reg, measurement[1:])

# ==========================================
# 4. 连接 IBM Quantum 并执行
# ==========================================
print(">>> 3. 连接 IBM Quantum 服务...")
try:
    service = QiskitRuntimeService(instance='open-instance')
except Exception:
    service = QiskitRuntimeService()

# 寻找后端
backend = service.least_busy(operational=True, simulator=False, min_num_qubits=10)
print(f"    已选定后端: {backend.name}")

# 转译
print(">>> 4. 正在转译电路...")
pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
isa_circuit = pm.run(circuit)

# 执行
print(">>> 5. 提交任务...")
sampler = Sampler(mode=backend)
job = sampler.run([isa_circuit], shots=4096)
print(f"    Job ID: {job.job_id()}")

# ==========================================
# 5. 结果分析 (核心修改：只统计 Ancilla=1)
# ==========================================
print("    正在等待结果...")
try:
    result = job.result()
    # SamplerV2 返回 bitstrings
    counts = result[0].data.meas.get_counts() 
    
    print("\n>>> 6. 后选择结果分析 (只保留 Ancilla=1 的数据):")
    
    filtered_counts = {}
    total_success_shots = 0
    
    # 遍历所有测量结果
    # 键(key)是二进制字符串，例如 "101 1" 或 "1011"
    # Qiskit 的经典寄存器顺序：[c_n ... c_1 c_0]
    # 我们定义了 meas[0] 是 ancilla (最右边)
    
    for bitstring, count in counts.items():
        # 清理空格（有些版本可能有空格分隔）
        clean_bitstring = bitstring.replace(" ", "")
        
        # 提取辅助比特 (最右边一位)
        ancilla_bit = clean_bitstring[-1] 
        
        # 提取解向量 (剩下的高位)
        solution_bits = clean_bitstring[:-1]
        
        # 筛选：只统计 ancilla 为 '1' 的情况
        if ancilla_bit == '1':
            total_success_shots += count
            # 累加到解向量分布中
            if solution_bits in filtered_counts:
                filtered_counts[solution_bits] += count
            else:
                filtered_counts[solution_bits] = count
                
    print(f"    总成功次数 (Ancilla=1): {total_success_shots} / 4096")
    print(f"    成功率: {total_success_shots/4096:.2%}")
    
    if total_success_shots > 0:
        print("\n    解向量分布 (归一化概率):")
        # 按概率从高到低排序
        sorted_results = sorted(filtered_counts.items(), key=lambda x: x[1], reverse=True)
        
        for state, count in sorted_results:
            prob = count / total_success_shots
            print(f"      |{state}> : {prob:.4f}")
    else:
        print("    [警告] 没有测量到任何 Ancilla=1 的结果。")

except Exception as e:
    print(f"获取结果时出错: {e}")




# (quantum) wangxinrui@Lenovo:~$ /home/wangxinrui/miniconda3/envs/quantum/bin/python /home/wangxinrui/Quantum-Computing/SRT/first_true.py
# >>> 1. 初始化数学参数...
#     t: 0.8767, C: 0.8179
# >>> 2. 构建量子电路...
# >>> 3. 连接 IBM Quantum 服务...
#     已选定后端: ibm_fez
# >>> 4. 正在转译电路...
# >>> 5. 提交任务...
#     Job ID: d4rftj45fjns73d1c92g
#     正在等待结果...

# >>> 6. 后选择结果分析 (只保留 Ancilla=1 的数据):
#     总成功次数 (Ancilla=1): 1558 / 4096
#     成功率: 38.04%

#     解向量分布 (归一化概率):
#       |000> : 0.1483
#       |001> : 0.1425
#       |100> : 0.1341
#       |101> : 0.1329
#       |111> : 0.1155
#       |010> : 0.1136
#       |110> : 0.1110
#       |011> : 0.1021