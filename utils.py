import os
import math
import multiprocessing
import psutil


def normalize_multipart_line_endings(text):
    r"""把 multipart 表单字段里的 CRLF 还原成 LF。

    multipart/form-data 的行尾是 CRLF，Node 的 FormData（undici）实测会把字段值里的
    `\n` 编码成 `\r\n`；Python / curl 客户端直传 `\n` 不发生这个转换。服务端按字面
    收下后，参考文本每行多出一个 `\r`，与调用方持有的原文不再逐字相等
    （2026-10-17 第 4 章：2136 字的原文被当成 2210 字，74 个换行各多一个字符）。

    在 HTTP 入口统一还原，参考文本与调用方逐字一致。孤立 `\r`（旧 Mac 行尾）
    同样按换行处理，避免它被下游的空白折叠规则变成一个普通空格。
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")


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