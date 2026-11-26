"""让测试文件能用 `from env_config import ...` 直接导入项目根下的模块。

业务代码是平铺布局（模块间互为平铺 import），没有 src/ 目录。测试收进 test/
后，pytest 只会把 test/ 放进 sys.path，项目根不在其中，直接跑 `pytest` 会报
ModuleNotFoundError；只有 `python -m pytest` 能跑通——那是因为它额外把当前
工作目录塞进了 sys.path，属于掩盖问题而非解决问题。此处显式补上项目根，
使 `pytest`、`uv run pytest`、以及从任意目录指定路径运行都一致。
"""

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
