from qiskit_ibm_runtime import QiskitRuntimeService
import numpy as np

# 1. 连接服务 (请确保已保存 API Key)
service = QiskitRuntimeService(instance='open-instance')

# 2. 填入你的 Job ID
job_id = 'd4rfehkfitbs739htu3g' 

# 3. 获取任务
print(f"正在获取任务 {job_id} 的状态...")
job = service.job(job_id)
status = job.status() # 获取状态字符串
print(f"当前状态: {status}")

# 4. 如果任务完成，提取结果
if status == 'DONE':
    print("任务已完成，正在下载结果...")
    result = job.result()
    pub_result = result[0]
    
    # 提取计数
    counts_ma = pub_result.data.ma.get_counts() # 辅助比特
    counts_c = pub_result.data.c.get_counts()   # 解向量
    
    print("\n--- 结果 ---")
    print(f"Ancilla (辅助比特) 计数: {counts_ma}")
    
    # 计算辅助比特为 1 的数量
    success_count = counts_ma.get('1', 0)
    print(f"成功筛选次数 (Ancilla=1): {success_count}")
    
    print(f"Solution (解向量) 计数: {counts_c}")

elif status == 'ERROR':
    print("任务执行出错！")
    print(job.error_message())

else:
    # 状态可能是 QUEUED (排队中) 或 RUNNING (运行中)
    print("任务尚未完成，请稍后再试。")
    print("提示：你可以在 IBM Quantum 网页控制台查看排队进度。")