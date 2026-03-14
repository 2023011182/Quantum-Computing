import numpy as np
import matplotlib.pyplot as plt

# Qiskit 核心
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit.library import QFTGate

# IBM Runtime
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# ==========================================
# 1. 连接 IBM Quantum
# ==========================================
print(">>> 1. 连接 IBM Quantum 服务...")
try:
    # 使用你之前确认的实例名称
    service = QiskitRuntimeService(instance='open-instance')
except Exception as e:
    print(f"连接失败，尝试默认连接: {e}")
    service = QiskitRuntimeService()

# 寻找最不繁忙的后端 (只需要 5 个量子比特就够了)
backend = service.least_busy(operational=True, simulator=False, min_num_qubits=5)
print(f"    已选定后端: {backend.name} ({backend.num_qubits} qubits)")

# ==========================================
# 2. 定义 2x2 HHL 电路参数
# ==========================================
print(">>> 2. 构建 2x2 HHL 电路 (手动构建哈密顿量演化)...")

# 矩阵 A = [[1, -0.5], [-0.5, 1]]
# 可以分解为 A = 1.0*I - 0.5*X
# 特征值约为 0.5 和 1.5
# 这是一个非常适合演示的小例子

# 参数设置
n_b = 1  # 向量 b 只需要 1 个量子比特 (2x2矩阵)
n_c = 3  # 时钟比特 3 个 (精度足够，深度可控)
n_a = 1  # 辅助比特 1 个
total_qubits = n_c + n_b + n_a

# 计算最佳演化时间 t
# 为了让特征值尽可能映射准确，我们需要仔细选择 t
# lambda_max ~ 1.5. 我们希望 1.5 * t < 2*pi
# 令 t = 2*pi * (3/4) / 1.5 = pi
t = np.pi 

# 寄存器
qr_clock = QuantumRegister(n_c, 'clock')
qr_b = QuantumRegister(n_b, 'b')
qr_ancilla = QuantumRegister(n_a, 'ancilla')
cr_measure = ClassicalRegister(n_b + 1, 'c') # 测量 b 和 ancilla

qc = QuantumCircuit(qr_ancilla, qr_clock, qr_b, cr_measure)

# --- A. 状态制备 (|b> = |1>) ---
qc.x(qr_b)  # 将 b 初始化为 |1>
qc.barrier()

# --- B. 量子相位估计 (QPE) ---
# 1. Clock 叠加态
qc.h(qr_clock)

# 2. 受控哈密顿量演化 U = e^{iAt}
# A = I - 0.5X
# e^{iAt} = e^{itI} * e^{-i 0.5 t X}
# e^{itI} 是全局相位，在受控操作中表现为 Phase Gate (Rz)
# e^{-i 0.5 t X} 是 RX 旋转，角度为 2 * (-0.5t) = -t
# 注意：受控 U^k 对应时间 t * 2^k

for i in range(n_c):
    time_step = t * (2 ** i)
    
    # 实现 Controlled-U(time_step)
    # U = e^{i * time_step * (I - 0.5X)}
    
    # 部分 1: e^{-i * 0.5 * time_step * X} -> RX 旋转
    # RX(theta) = e^{-i * theta/2 * X} => theta/2 = 0.5*time_step => theta = time_step
    # 注意 Qiskit 的 RX 定义，我们需要旋转 -time_step
    qc.crx(-time_step, qr_clock[i], qr_b[0])
    
    # 部分 2: e^{i * time_step * I} -> 全局相位变为受控相位 (P gate / U1)
    qc.p(time_step, qr_clock[i])

# 3. 逆 QFT
qc.append(QFTGate(n_c).inverse(), qr_clock)
qc.barrier()

# --- C. 受控旋转 (Eigenvalue Inversion) ---
# 这里的 C_const 是 HHL 的辅助常数
C_const = 0.5 

# 对每个可能的时钟状态 |k> 进行受控旋转
# k 代表特征值 lambda = k * (2*pi) / (t * 2^n_c)
for k in range(1, 2**n_c):
    k_bin = format(k, f'0{n_c}b')[::-1] # Little-endian
    
    # 计算对应的 lambda
    lam = k * (2 * np.pi) / (t * (2**n_c))
    
    # 避免除零
    if lam < 1e-6: continue
    
    # 计算旋转角度
    ratio = C_const / lam
    if ratio > 1: ratio = 1
    if ratio < -1: ratio = -1
    theta = 2 * np.arcsin(ratio)
    
    # 构建多控旋转
    # 1. 激活控制位 (X门)
    for i, bit in enumerate(k_bin):
        if bit == '0': qc.x(qr_clock[i])
            
    # 2. 执行受控 RY
    # 使用 mcry (Multi-Controlled RY)
    qc.mcry(theta, qr_clock, qr_ancilla[0])
    
    # 3. 恢复控制位 (X门)
    for i, bit in enumerate(k_bin):
        if bit == '0': qc.x(qr_clock[i])

qc.barrier()

# --- D. 逆 QPE (Uncomputation) ---
qc.append(QFTGate(n_c), qr_clock)

for i in range(n_c-1, -1, -1):
    time_step = t * (2 ** i)
    qc.p(-time_step, qr_clock[i]) # 逆相位
    qc.crx(time_step, qr_clock[i], qr_b[0]) # 逆旋转 (角度取反)

qc.h(qr_clock)
qc.barrier()

# --- E. 测量 ---
# 测量 Ancilla (作为标志位) 和 b (作为解)
qc.measure(qr_ancilla, cr_measure[0])
qc.measure(qr_b, cr_measure[1])

# ==========================================
# 3. 转译与执行
# ==========================================
print(">>> 3. 转译电路 (这次应该会快很多)...")
# 优化等级设为 3，尽可能减少深度
pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
isa_circuit = pm.run(qc)

# 打印电路统计
depth = isa_circuit.depth()
# 统计 2-qubit 门 (ECR/CX/CZ)
ops = isa_circuit.count_ops()
num_2q = ops.get('ecr', 0) + ops.get('cx', 0) + ops.get('cz', 0)

print(f"    物理电路深度: {depth}")
print(f"    2-qubit 门数量: {num_2q}")
print(f"    (对比之前的 37,000 层，这个是可以运行的！)")

print(">>> 4. 提交任务...")
sampler = Sampler(mode=backend)
job = sampler.run([isa_circuit], shots=4096)
print(f"    Job ID: {job.job_id()}")

# ==========================================
# 4. 获取结果
# ==========================================
print("    正在等待结果 (请耐心等待，不要中断)...")
try:
    result = job.result()
    print(">>> 5. 任务完成！")
    
    pub_result = result[0]
    counts = pub_result.data.c.get_counts()
    
    print("\n--- 原始测量结果 (Key格式: 'b a', a是低位) ---")
    # 注意：SamplerV2 测量结果通常是 BitString，这里 cr_measure 有 2 位
    # 格式可能是 "b_val ancilla_val" 或者合并的 "ba"
    # 根据 QuantumCircuit 定义: measure(ancilla, c[0]), measure(b, c[1])
    # Qiskit 也是 Little-endian，c[0] 是最右边
    
    # 筛选 Ancilla = 1 的结果
    # 我们需要的格式是 "11" (b=1, a=1) 和 "01" (b=0, a=1)
    # 因为 c[0] 是 ancilla
    
    count_0_1 = 0 # b=0, a=1
    count_1_1 = 0 # b=1, a=1
    
    for key, val in counts.items():
        # key 是一串二进制，我们需要解析
        # 假设 key 长度对应 classical bits 数量
        # c[0] 是 ancilla (最右边), c[1] 是 b (左边)
        # 例如 '10' -> b=1, ancilla=0
        # 我们只关心以 '1' 结尾的 (ancilla=1)
        
        # 处理可能的空格
        key_clean = key.replace(" ", "")
        
        if key_clean.endswith('1'): # Ancilla 测量成功
            if key_clean.startswith('0'):
                count_0_1 += val
            elif key_clean.startswith('1'):
                count_1_1 += val
                
    total_success = count_0_1 + count_1_1
    
    print(f"\n后选择成功次数 (Ancilla=1): {total_success}")
    
    if total_success > 0:
        prob_0 = count_0_1 / total_success
        prob_1 = count_1_1 / total_success
        print(f"解向量分布:")
        print(f"  |0> : {prob_0:.4f}")
        print(f"  |1> : {prob_1:.4f}")
        
        # 理论解验证
        # A = [[1, -0.5], [-0.5, 1]], b = |1> = [0, 1]
        # x = A^-1 b
        # A^-1 = [[1.33, 0.66], [0.66, 1.33]]
        # x = [0.66, 1.33]
        # 归一化后概率: |0> ~ 0.2, |1> ~ 0.8
        print("\n理论参考:")
        print("  |0> : ~0.2000")
        print("  |1> : ~0.8000")
    else:
        print("未能测量到 Ancilla=1 的结果 (这就是 HHL 的概率性挑战)")

except Exception as e:
    print(f"获取结果时出错: {e}")

# (quantum) wangxinrui@Lenovo:~$ /home/wangxinrui/miniconda3/envs/quantum/bin/python /home/wangxinrui/Quantum-Computing/SRT/try.py
# >>> 1. 连接 IBM Quantum 服务...
#     已选定后端: ibm_fez (156 qubits)
# >>> 2. 构建 2x2 HHL 电路 (手动构建哈密顿量演化)...
# >>> 3. 转译电路 (这次应该会快很多)...
#     物理电路深度: 677
#     2-qubit 门数量: 198
#     (对比之前的 37,000 层，这个是可以运行的！)
# >>> 4. 提交任务...
#     Job ID: d4rfc8vt3pms73985ung
#     正在等待结果 (请耐心等待，不要中断)...
# >>> 5. 任务完成！

# --- 原始测量结果 (Key格式: 'b a', a是低位) ---

# 后选择成功次数 (Ancilla=1): 2024
# 解向量分布:
#   |0> : 0.3612
#   |1> : 0.6388

# 理论参考:
#   |0> : ~0.2000
#   |1> : ~0.8000