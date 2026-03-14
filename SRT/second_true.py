import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
import itertools # 用于生成所有可能的态

# Qiskit 库
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile
from qiskit.circuit.library import QFTGate, UnitaryGate
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# ==========================================
# 1. 定义矩阵 A 和向量 b (保持不变)
# ==========================================
print(">>> 1. 初始化数学参数...")
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
matrix_A = np.block([[zero_8x8, M], [M.T, zero_8x8]])

vector_b = np.array([0.2625, 0.4375, 0.35, 0.175, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
normalized_b = vector_b / np.linalg.norm(vector_b)

# ==========================================
# 2. 设置 HHL 参数
# ==========================================
n_b = 4 
n_c = 6 
n_a = 1 

eigenvalues = np.linalg.eigvalsh(matrix_A)
lambda_max_abs = np.max(np.abs(eigenvalues))
# t 的选择：避免相位溢出
t = (np.pi * 0.9) / lambda_max_abs
# C 的选择：保证旋转角度有效
C_val = np.min(np.abs(eigenvalues)) * 0.99

print(f"    t: {t:.4f}, C: {C_val:.4f}")

# ==========================================
# 3. 构建量子电路
# ==========================================
print(">>> 2. 构建量子电路...")
clock = QuantumRegister(n_c, name='clock')
input_reg = QuantumRegister(n_b, name='b')
ancilla = QuantumRegister(n_a, name='ancilla')
meas_all = ClassicalRegister(n_b + 1, name='meas_all')

circuit = QuantumCircuit(ancilla, clock, input_reg, meas_all)

# (A) 状态准备
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

# (C) 逆 QFT
circuit.append(QFTGate(num_qubits=n_c).inverse(), clock)
circuit.barrier()

# (D) 旋转
for k in range(1, 2**n_c):
    if k < 2**(n_c - 1): k_signed = k 
    else: k_signed = k - 2**n_c 
    lambda_val = (2 * np.pi * k_signed) / (t * (2**n_c))
    if abs(lambda_val) < 1e-6: continue
    ratio = C_val / lambda_val
    ratio = max(-1.0, min(1.0, ratio))
    theta = 2 * np.arcsin(ratio)
    k_binary = format(k, f'0{n_c}b')[::-1]
    for idx, bit in enumerate(k_binary):
        if bit == '0': circuit.x(clock[idx])
    circuit.mcry(theta, clock[:], ancilla[0])
    for idx, bit in enumerate(k_binary):
        if bit == '0': circuit.x(clock[idx])
circuit.barrier()

# (E) 逆 QPE
circuit.append(QFTGate(num_qubits=n_c), clock)
for k in reversed(range(n_c)):
    power = 2**k
    exponent = -1j * matrix_A * t * power 
    u_matrix_dag = expm(exponent)
    u_gate_dag = UnitaryGate(u_matrix_dag, label=f"U_dag^{power}")
    circuit.append(u_gate_dag.control(1), [clock[k]] + list(input_reg))
circuit.h(clock)
circuit.barrier()

# (F) 测量
circuit.measure(ancilla[0], meas_all[0])
circuit.measure(input_reg, meas_all[1:])

# ==========================================
# 4. 连接硬件并执行
# ==========================================
print(">>> 3. 连接 IBM Quantum 服务...")
try:
    service = QiskitRuntimeService(instance='open-instance')
    print(f"    成功连接: {service.active_account()['instance']}")
except Exception:
    print("    尝试默认连接...")
    service = QiskitRuntimeService()

# 寻找后端
print("    正在寻找后端...")
try:
    backend = service.least_busy(operational=True, simulator=False, min_num_qubits=11)
except Exception:
    print("    [警告] 无可用真机，切换到模拟器...")
    from qiskit_aer import AerSimulator
    backend = AerSimulator()

print(f"    已选定后端: {backend.name}")

# 转译
print(">>> 4. 正在转译电路...")
pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
isa_circuit = pm.run(circuit)

depth = isa_circuit.depth()
ops = isa_circuit.count_ops()
num_2q = ops.get('ecr', 0) + ops.get('cx', 0) + ops.get('cz', 0)
print(f"    物理电路深度: {depth}")
print(f"    2-qubit 门数量: {num_2q}")

# 执行
TOTAL_SHOTS = 50000
print(f">>> 5. 提交任务 (Shots={TOTAL_SHOTS})...")

try:
    sampler = Sampler(mode=backend)
    job = sampler.run([isa_circuit], shots=TOTAL_SHOTS)
    print(f"    Job ID: {job.job_id()}")
    
    print("    正在等待结果...")
    result = job.result()
    print(">>> 6. 任务完成！")

    # ==========================================
    # 5. 后选择结果分析 (全量展示)
    # ==========================================
    pub_result = result[0]
    counts = pub_result.data.meas_all.get_counts()
    
    print("\n>>> 7. 详细结果对比 (Ancilla=1):")
    
    # 1. 统计量子实验数据
    filtered_counts = {}
    success_shots = 0
    
    for bitstring, count in counts.items():
        bit_str = bitstring.replace(" ", "")
        ancilla_bit = bit_str[-1]
        solution_bits = bit_str[:-1]
        
        if ancilla_bit == '1':
            success_shots += count
            if solution_bits in filtered_counts:
                filtered_counts[solution_bits] += count
            else:
                filtered_counts[solution_bits] = count
    
    print(f"    总运行次数: {TOTAL_SHOTS}")
    print(f"    成功筛选次数: {success_shots}")
    print(f"    成功率: {success_shots/TOTAL_SHOTS:.2%}")
    
    # 2. 计算经典理论解
    try:
        x_exact = np.linalg.solve(matrix_A, vector_b)
        probs_exact = np.abs(x_exact / np.linalg.norm(x_exact))**2
    except Exception as e:
        print(f"    经典解计算失败: {e}")
        probs_exact = np.zeros(2**n_b)

    # 3. 打印对比表格 (所有 16 个态)
    print("\n    {:<10} {:<15} {:<15} {:<10}".format("State", "Quantum Prob", "Classic Prob", "Diff"))
    print("    " + "-"*50)
    
    # 生成所有可能的 4 位二进制态 ('0000', '0001', ..., '1111')
    all_states = ["".join(seq) for seq in itertools.product("01", repeat=n_b)]
    
    # 将态按照经典概率从大到小排序，方便观察主要成分
    # 创建一个 (state, classic_prob) 的列表
    states_with_probs = []
    for i, state in enumerate(all_states):
        # 注意：这里假设 numpy 数组索引 i 对应二进制 state 的整数值
        # 例如 i=0 -> '0000', i=1 -> '0001' (Big-endian for display)
        # Qiskit 默认是 Little-endian，但 format(i, '04b') 是标准的 Big-endian 写法
        # 只要我们 consistent 即可。通常 solver 结果 x[0] 对应 |0000>
        states_with_probs.append((state, probs_exact[i]))
        
    # 排序
    states_with_probs.sort(key=lambda x: x[1], reverse=True)
    
    for state, classic_prob in states_with_probs:
        # 获取量子统计次数 (如果没有测到则为 0)
        q_count = filtered_counts.get(state, 0)
        
        # 计算量子概率 (避免除以零)
        quantum_prob = q_count / success_shots if success_shots > 0 else 0.0
        
        diff = quantum_prob - classic_prob
        
        print(f"    |{state}>   {quantum_prob:.4f}          {classic_prob:.4f}          {diff:+.4f}")

except Exception as e:
    print(f"执行过程中出错: {e}")


# (quantum) wangxinrui@Lenovo:~$ /home/wangxinrui/miniconda3/envs/quantum/bin/python /home/wangxinrui/Quantum-Computing/SRT/second_true.py
# >>> 1. 初始化数学参数...
#     t: 0.4383, C: 0.8179
# >>> 2. 构建量子电路...
# >>> 3. 连接 IBM Quantum 服务...
#     成功连接: crn:v1:bluemix:public:quantum-computing:us-east:a/3467414db4e14356b0044b7e985469eb:9948e9b4-786c-4d84-8a9c-69fc8fc57fd5::
#     正在寻找后端...
#     已选定后端: ibm_fez
# >>> 4. 正在转译电路...
#     物理电路深度: 39736
#     2-qubit 门数量: 16561
# >>> 5. 提交任务 (Shots=50000)...
#     Job ID: d4rg6bjher1c73bc8jj0
#     正在等待结果...
# >>> 6. 任务完成！

# >>> 7. 详细结果对比 (Ancilla=1):
#     总运行次数: 50000
#     成功筛选次数: 6524
#     成功率: 13.05%

#     State      Quantum Prob    Classic Prob    Diff      
#     --------------------------------------------------
#     |1001>   0.0674          0.2155          -0.1481
#     |1011>   0.0697          0.2155          -0.1458
#     |1101>   0.0717          0.2155          -0.1438
#     |1111>   0.0710          0.2155          -0.1445
#     |1000>   0.0561          0.0345          +0.0216
#     |1010>   0.0584          0.0345          +0.0239
#     |1100>   0.0621          0.0345          +0.0276
#     |1110>   0.0567          0.0345          +0.0222
#     |0000>   0.0544          0.0000          +0.0544
#     |0001>   0.0630          0.0000          +0.0630
#     |0010>   0.0578          0.0000          +0.0578
#     |0011>   0.0682          0.0000          +0.0682
#     |0100>   0.0567          0.0000          +0.0567
#     |0101>   0.0656          0.0000          +0.0656
#     |0110>   0.0533          0.0000          +0.0533
#     |0111>   0.0677          0.0000          +0.0677