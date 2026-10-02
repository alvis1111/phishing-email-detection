"""真实数据加载器。

内置一个可下载的公开钓鱼邮件数据集（Zenodo），并支持从任意 CSV
自动识别「文本列 + 标签列」，把数据转成统一格式：

    {"sender": "", "subject": "", "body": 文本, "label": 0/1}

标签映射：包含 phishing / spam / fraud / scam 记为 1（钓鱼/恶意），其余记为 0。
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

import config
from data_synthesis import generate_dataset

# Zenodo: Phishing Validation Emails Dataset（2000 封，Safe / Phishing 各 1000）
REAL_DATA_URL = "https://zenodo.org/api/records/13474746/files/Phishing_validation_emails.csv/content"
REAL_DATA_PATH = config.DATA_DIR / "Phishing_validation_emails.csv"

_TEXT_HINTS = ["text", "body", "content", "message", "email"]
_LABEL_HINTS = ["label", "type", "class", "category"]
_POSITIVE_HINTS = ["phishing", "spam", "fraud", "scam"]


def download_real_dataset(path: Path | str = REAL_DATA_PATH, force: bool = False) -> Path:
    """下载 Zenodo 公开数据集（幂等，已存在则跳过）。"""
    path = Path(path)
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"下载真实数据集到 {path} ...")
    urllib.request.urlretrieve(REAL_DATA_URL, path)
    print("下载完成。")
    return path


def _find_text_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if any(k in c.lower() for k in _TEXT_HINTS):
            return c
    for c in df.columns:
        if df[c].dtype == object:
            return c
    return df.columns[0]


def _find_label_col(df: pd.DataFrame, text_col: str) -> str:
    for c in df.columns:
        if c != text_col and any(k in c.lower() for k in _LABEL_HINTS):
            return c
    return df.columns[-1]


def _to_label(v) -> int:
    if isinstance(v, (int, float, bool)):
        return int(bool(v))
    s = str(v).lower()
    return int(any(k in s for k in _POSITIVE_HINTS))


def records_from_df(df: pd.DataFrame, text_col: str | None = None, label_col: str | None = None) -> list[dict]:
    """把 DataFrame 转成统一邮件记录列表（含原始头部解析）。"""
    from features import parse_raw_email

    tc = text_col or _find_text_col(df)
    lc = label_col or _find_label_col(df, tc)
    records = []
    for t, l in zip(df[tc], df[lc]):
        parsed = parse_raw_email(str(t))
        records.append(
            {"sender": parsed["sender"], "subject": parsed["subject"], "body": parsed["body"], "label": _to_label(l)}
        )
    return records


def load_real_records(path: Path | str = REAL_DATA_PATH, dedup: bool = True) -> list[dict]:
    """加载内置 Zenodo 数据集。

    该数据 2000 行里只有 100 个唯一文本（模板化严重），不先去重的话，
    交叉验证会把同一个模板同时放进训练/测试，导致指标虚高。因此默认去重。
    """
    download_real_dataset(path)
    df = pd.read_csv(path)
    if dedup:
        before = len(df)
        df = df.drop_duplicates(subset=["Email Text"]).reset_index(drop=True)
        print(f"真实数据去重：{before} -> {len(df)} 条（原数据模板化，仅 {len(df)} 个唯一文本）")
    return records_from_df(df, text_col="Email Text", label_col="Email Type")


# Kaggle 数据集默认句柄：naserabdullahalam/phishing-email-dataset（约 8.2 万封）
KAGGLE_HANDLE = "naserabdullahalam/phishing-email-dataset"
KAGGLE_CSV = "phishing_email.csv"


def download_kaggle_dataset(handle: str = KAGGLE_HANDLE) -> Path:
    """通过 kagglehub 下载 Kaggle 数据集（公开数据可匿名下载，无需 token）。

    若网络不稳定导致下载中断，可用 curl 断点续传：
        curl -L -C - "https://www.kaggle.com/api/v1/datasets/download/<handle>" -o data/phishing_email.zip
    解压后把 phishing_email.csv 放到 data/ 下，load_kaggle_records 会自动优先使用本地文件。
    """
    try:
        import kagglehub
    except ImportError:
        raise SystemExit("[错误] 需要 kagglehub，请执行: pip install kagglehub")
    print(f"下载 Kaggle 数据集 {handle}（首次约 77MB）...")
    p = kagglehub.dataset_download(handle)
    print("下载完成。")
    return Path(p)


def _find_kaggle_csv() -> Path | None:
    """优先找本地已下载/解压的 CSV，避免重复下载。"""
    local = config.DATA_DIR / KAGGLE_CSV
    if local.exists():
        return local
    cache = Path.home() / ".cache" / "kagglehub" / "datasets"
    if cache.exists():
        matches = list(cache.glob(f"**/{KAGGLE_CSV}"))
        if matches:
            return matches[0]
    return None


def load_kaggle_records(handle: str = KAGGLE_HANDLE, n: int | None = None, seed: int = config.SEED) -> list[dict]:
    """加载 Kaggle phishing_email 数据集（text_combined + label）。

    用 parse_raw_email 把 text_combined 拆成 sender/subject/body，
    让发件人、正文、URL 三类特征都能用上。
    n 不为空时随机抽样 n 条，便于快速验证。
    """
    from features import parse_raw_email

    csv = _find_kaggle_csv()
    if csv is None:
        d = download_kaggle_dataset(handle)
        csvs = list(d.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"未在 {d} 找到 CSV 文件")
        csv = csvs[0]
    print(f"读取 {csv.name} ...")

    header = pd.read_csv(csv, nrows=0)
    tc = _find_text_col(header)
    lc = _find_label_col(header, tc)
    print(f"识别到文本列: {tc}，标签列: {lc}")

    df = pd.read_csv(csv, usecols=[tc, lc])
    if n:
        df = df.sample(n=n, random_state=seed).reset_index(drop=True)

    records = []
    for t, l in zip(df[tc], df[lc]):
        parsed = parse_raw_email(str(t))
        records.append(
            {"sender": parsed["sender"], "subject": parsed["subject"], "body": parsed["body"], "label": _to_label(l)}
        )
    return records


# 带原始正文/URL 的 Kaggle 18k 数据集（Email Text 含正文+URL+部分 From 头）
KAGGLE_RAW_HANDLE = "subhajournal/phishingemails"
KAGGLE_RAW_CSV = "Phishing_Email.csv"


def download_kaggle_raw_dataset(handle: str = KAGGLE_RAW_HANDLE) -> Path:
    """下载 subhajournal/phishingemails（优先用本地已有的 CSV）。"""
    local = config.DATA_DIR / KAGGLE_RAW_CSV
    if local.exists():
        return local
    d = download_kaggle_dataset(handle)
    csvs = list(d.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"未在 {d} 找到 CSV 文件")
    return csvs[0]


def load_raw_records(n: int | None = None, seed: int = config.SEED) -> list[dict]:
    """加载带原始正文/URL 的 Kaggle 18k 数据集。

    Email Text 里部分邮件含 From/Subject 头，多数含真实 URL 与邮箱地址；
    parse_raw_email 会解析头部，并在无 From 头时把正文里第一个邮箱地址作为发件人兜底。
    """
    csv = download_kaggle_raw_dataset()
    print(f"读取 {csv.name} ...")
    header = pd.read_csv(csv, nrows=0)
    tc = _find_text_col(header)
    lc = _find_label_col(header, tc)
    print(f"识别到文本列: {tc}，标签列: {lc}")
    df = pd.read_csv(csv, usecols=[tc, lc])
    if n:
        df = df.sample(n=n, random_state=seed).reset_index(drop=True)
    return records_from_df(df, text_col=tc, label_col=lc)


def load_records(
    source: str = "synthetic",
    n_normal: int = 250,
    n_phishing: int = 250,
    seed: int = config.SEED,
    dedup: bool = True,
    sample: int | None = None,
) -> list[dict]:
    """按来源加载邮件记录。

    source 取值：
        "synthetic"  -> 合成演示数据
        "real"       -> 内置 Zenodo 公开数据集
        "kaggle"     -> Kaggle phishing_email 数据集（约 8.2 万封，预处理文本）
        "raw"        -> Kaggle 18k 数据集（含原始正文/URL/部分头部）
        其它         -> 视为本地 CSV 路径（自动识别文本/标签列，含头部解析）
    """
    if source == "synthetic":
        return generate_dataset(n_normal, n_phishing, seed)
    if source == "real":
        return load_real_records(dedup=dedup)
    if source == "kaggle":
        return load_kaggle_records(n=sample, seed=seed)
    if source == "raw":
        return load_raw_records(n=sample, seed=seed)
    df = pd.read_csv(source)
    return records_from_df(df)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="下载内置真实数据集")
    parser.add_argument("--no-dedup", action="store_true", help="查看不去重时的完整数据量")
    args = parser.parse_args()

    download_real_dataset()
    recs = load_real_records(dedup=not args.no_dedup)
    n_phish = sum(r["label"] for r in recs)
    print(f"共 {len(recs)} 条：钓鱼 {n_phish} / 正常 {len(recs) - n_phish}")
