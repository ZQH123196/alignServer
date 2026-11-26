import os
import math
import multiprocessing
import psutil

# ==========================================
# 0. 容器资源获取辅助函数 (已修订探针优先级)
# ==========================================
def get_container_cpu_count():
    """获取容器实际分配的 CPU 核心数 (优先读取 cgroup 限制，防止 affinity 穿透到宿主机)"""
    cgroup_cores = None
    try:
        # cgroups v2 限制读取
        if os.path.exists('/sys/fs/cgroup/cpu.max'):
            with open('/sys/fs/cgroup/cpu.max') as f:
                quota, period = f.read().strip().split()
                if quota != 'max':
                    cgroup_cores = math.ceil(int(quota) / int(period))
        # cgroups v1 限制读取
        elif os.path.exists('/sys/fs/cgroup/cpu/cpu.cfs_quota_us'):
            with open('/sys/fs/cgroup/cpu/cpu.cfs_quota_us') as f:
                quota = int(f.read().strip())
            with open('/sys/fs/cgroup/cpu/cpu.cfs_period_us') as f:
                period = int(f.read().strip())
            if quota > 0:
                cgroup_cores = math.ceil(quota / period)
    except Exception:
        pass
        
    # 获取进程亲和度核心数或宿主机总核心数作为对比上限
    affinity_cores = multiprocessing.cpu_count()
    try:
        affinity_cores = len(os.sched_getaffinity(0))
    except AttributeError:
        pass

    # 如果 cgroup 设定了配额，取配额和亲和核心数的最小值（防止云机共享带来的穿透误判）
    if cgroup_cores is not None:
        return min(cgroup_cores, affinity_cores)
    return affinity_cores

def get_container_available_memory():
    """获取容器实际可用的内存 (Bytes)"""
    host_mem = psutil.virtual_memory().available
    cgroup_limit = None
    cgroup_usage = 0

    try:
        # cgroups v2
        if os.path.exists('/sys/fs/cgroup/memory.max'):
            with open('/sys/fs/cgroup/memory.max') as f:
                val = f.read().strip()
                if val != 'max':
                    cgroup_limit = int(val)
            if os.path.exists('/sys/fs/cgroup/memory.current'):
                with open('/sys/fs/cgroup/memory.current') as f:
                    cgroup_usage = int(f.read().strip())
        
        # cgroups v1
        elif os.path.exists('/sys/fs/cgroup/memory/memory.limit_in_bytes'):
            with open('/sys/fs/cgroup/memory/memory.limit_in_bytes') as f:
                val = int(f.read().strip())
                if val < 9223372036854771712:
                    cgroup_limit = val
            if os.path.exists('/sys/fs/cgroup/memory/memory.usage_in_bytes'):
                with open('/sys/fs/cgroup/memory/memory.usage_in_bytes') as f:
                    cgroup_usage = int(f.read().strip())
                    
        if cgroup_limit is not None:
            return min(host_mem, cgroup_limit - cgroup_usage)
    except Exception:
        pass

    return host_mem