"""全局配置：路径与常量。"""
import sys
from pathlib import Path


def _setup_utf8_console() -> None:
    """把标准输出/错误流切到 UTF-8，避免 Windows 控制台中文乱码。

    旧版 cmd/PowerShell 若仍显示乱码，可先执行 `chcp 65001` 切到 UTF-8 代码页。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


_setup_utf8_console()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
PLOTS_DIR = BASE_DIR / "plots"

DATA_PATH = DATA_DIR / "emails.csv"
MODEL_PATH = MODEL_DIR / "model_artifact.joblib"

SEED = 42
