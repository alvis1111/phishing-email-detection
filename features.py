"""特征提取。

从一封邮件（dict: sender/subject/body）中抽取三类特征：
    1. 发件人特征（前缀 snd_）
    2. 正文特征（前缀 body_）
    3. URL 特征（前缀 url_）

最终合并成一个固定键集合的特征向量，训练与预测共用同一套提取逻辑，
保证特征列一致。
"""
from __future__ import annotations

import re
from urllib.parse import urlsplit

import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer

# 品牌 -> 官方域名，用于识别仿冒域名（键为小写）
BRANDS = {
    "paypal": "paypal.com",
    "apple": "apple.com",
    "microsoft": "microsoft.com",
    "amazon": "amazon.com",
    "netflix": "netflix.com",
    "google": "google.com",
    "chase": "chase.com",
    "bankofamerica": "bankofamerica.com",
    "wellsfargo": "wellsfargo.com",
    "ebay": "ebay.com",
    "dropbox": "dropbox.com",
    "linkedin": "linkedin.com",
    "dhl": "dhl.com",
    "fedex": "fedex.com",
    "ups": "ups.com",
    "usps": "usps.com",
    "irs": "irs.gov",
}

# 品牌关联的合法域名白名单（官方主域之外的品牌方运营域）。
# 这些域名不应被判为「仿冒」或「显示名与邮箱域名不符」，
# 用于降低对正规品牌营销邮件（如 microsoftrewards.com）的误报。
# 注意：故意不含 amazonaws.com 等常被钓鱼滥用的域。
BRAND_WHITELIST = {
    "microsoft": ["microsoftrewards.com", "microsoftonline.com", "microsoftstore.com", "office.com", "live.com", "outlook.com"],
    "paypal": ["paypal.me", "paypal-communication.com", "paypalobjects.com"],
    "apple": ["icloud.com", "itunes.com"],
    "amazon": ["amazonpay.com", "media-amazon.com"],
    "google": ["googlemail.com", "youtube.com", "googleusercontent.com"],
    "netflix": ["netflix.net"],
    "linkedin": ["lnkd.in"],
    "dropbox": ["dropboxmail.com"],
    "chase": ["jpmorgan.com", "chasepaymentech.com"],
    "bankofamerica": ["bofa.com"],
}

FREE_EMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com",
    "live.com", "mail.ru", "protonmail.com", "qq.com", "163.com",
}

SUSPICIOUS_TLDS = {
    "xyz", "top", "club", "biz", "info", "click", "link", "tk", "ml",
    "ga", "cf", "gq", "work", "zip", "online", "site", "icu", "rest",
}

URGENCY_WORDS = ["urgent", "immediately", "immediate", "asap", "act now", "right now", "act quickly", "quickly"]
TIME_PRESSURE = [
    "24 hour", "24 hours", "48 hour", "48 hours", "within 24", "within 48",
    "expires", "expire", "deadline", "deadlines", "limited time",
]
CREDENTIAL_WORDS = [
    "password", "passwords", "credential", "credentials", "login", "logins",
    "sign in", "sign-in", "ssn", "social security", "card number", "card numbers",
]
THREAT_WORDS = [
    "suspend", "suspended", "suspension",
    "terminate", "terminated", "termination",
    "delete", "deleted",
    "lock", "locked",
    "restrict", "restricted",
    "close", "closed",
    "disable", "disabled",
]
PRIZE_WORDS = [
    "won", "winner", "winners", "prize", "prizes", "reward", "rewards",
    "congratulations", "lottery", "gift card", "gift cards", "free gift", "free gifts", "claim",
]
SUSPICIOUS_PATH = {"verify", "login", "confirm", "secure", "unlock", "validate", "recover", "signin", "update"}

URL_RE = re.compile(r'https?://[^\s<>"\'()]+')
EMAIL_RE = re.compile(r'<([^>]+)>')
IPV4_RE = re.compile(r'^(\d{1,3}\.){3}\d{1,3}$')
BARE_EMAIL_RE = re.compile(r'[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}')
HEADER_RE = re.compile(r'^(from|sender|subject|date|to|reply-to|return-path)\s*:\s*(.+)$', re.IGNORECASE)


def parse_raw_email(text: str) -> dict:
    """从原始邮件文本（可能含 From:/Subject: 头）中拆出 sender / subject / body。

    返回 {"sender": ..., "subject": ..., "body": ...}。若没有头部信息，
    sender/subject 置空、整段文本作为 body。
    """
    sender, subject = "", ""
    lines = text.splitlines()
    header_end = -1
    seen_blank = False
    for i, raw in enumerate(lines):
        line = raw.strip()
        m = HEADER_RE.match(line)
        if m:
            key, val = m.group(1).lower(), m.group(2).strip()
            if key in ("from", "sender", "return-path") and not sender:
                sender = val
            elif key == "subject" and not subject:
                subject = val
            header_end = i
        elif line == "":
            seen_blank = True
            if header_end >= 0:
                # 头部之后遇到空行，之后的都是正文
                break
        else:
            if seen_blank or header_end >= 0 and i > header_end + 1:
                pass
    # 头部区（含其后的空行）之后的部分；若没解析出头部，则全文作为正文
    if header_end >= 0:
        body_start = header_end + 1
        while body_start < len(lines) and lines[body_start].strip() == "":
            body_start += 1
        body = "\n".join(lines[body_start:]).strip()
    else:
        body = text.strip()

    # sender 直接返回 From/Sender 头的原始值；_parse_sender 能同时处理
    # "Name <email>" 与 "bare@email" 两种格式，无需在此二次包装。
    # 若没解析出 sender（无 From/Sender 头），从全文找第一个邮箱地址作为兜底，
    # 让发件人特征在只有正文邮箱的邮件上也能启用（启发式，非严格 From 字段）。
    if not sender:
        m = BARE_EMAIL_RE.search(text)
        if m:
            sender = m.group(0)
    return {"sender": sender, "subject": subject, "body": body}


def _parse_sender(sender: str) -> tuple[str, str]:
    """返回 (显示名, 邮箱地址)。缺失时用空串。"""
    display, email = "", sender.strip()
    m = EMAIL_RE.search(sender)
    if m:
        email = m.group(1).strip()
        display = sender[: m.start()].strip()
    return display, email


def _split_email(email: str) -> tuple[str, str]:
    if "@" not in email:
        return "", ""
    local, domain = email.rsplit("@", 1)
    return local.lower(), domain.lower()


def _contains_brand(s: str) -> list[str]:
    """返回 s 中包含的品牌名（小写），用词边界避免误匹配（如 purchase 中的 chase）。"""
    s = s.lower()
    return [b for b in BRANDS if re.search(rf"\b{re.escape(b)}\b", s)]


def _is_brand_domain_ok(domain: str, brand: str) -> bool:
    """domain 是否为 brand 的官方主域或白名单中的合法品牌域（含子域）。"""
    d = domain.lower()
    official = BRANDS[brand]
    if d == official or d.endswith("." + official):
        return True
    return any(d == w or d.endswith("." + w) for w in BRAND_WHITELIST.get(brand, []))


def _lookalike(domain: str) -> int:
    """域名是否包含品牌名但并非该品牌的合法域名（仿冒）。"""
    d = domain.lower()
    for brand in BRANDS:
        if re.search(rf"\b{re.escape(brand)}\b", d):
            if _is_brand_domain_ok(d, brand):
                return 0
            return 1
    return 0


def extract_sender_features(sender: str) -> dict:
    display, email = _parse_sender(sender)
    local, domain = _split_email(email)
    brands_in_display = _contains_brand(display)

    return {
        "snd_is_free_email": int(domain in FREE_EMAIL_DOMAINS),
        "snd_domain_len": len(domain),
        "snd_domain_digits": sum(c.isdigit() for c in domain),
        "snd_domain_hyphens": domain.count("-"),
        "snd_localpart_len": len(local),
        "snd_localpart_digits": sum(c.isdigit() for c in local),
        "snd_lookalike_brand": _lookalike(domain),
        # 显示名提到品牌，但邮箱域名与官方/白名单不符 => 伪造发件人
        "snd_display_brand_mismatch": int(
            bool(brands_in_display)
            and not any(_is_brand_domain_ok(domain, b) for b in brands_in_display)
        ),
    }


def _extract_urls(body: str) -> list[str]:
    return URL_RE.findall(body)


def _has_any_word(text: str, words: list[str]) -> bool:
    """词边界匹配，避免子串误报（如 unlocked 命中 locked、wonder 命中 won）。"""
    return any(re.search(rf"\b{re.escape(w)}\b", text) for w in words)


def extract_body_features(body: str) -> dict:
    words = body.split()
    n_chars = len(body)
    n_letters = sum(c.isalpha() for c in body)
    n_upper = sum(c.isupper() for c in body)
    lower = body.lower()

    return {
        "body_len": n_chars,
        "body_word_count": len(words),
        "body_caps_ratio": (n_upper / n_letters) if n_letters else 0.0,
        "body_exclamation_count": body.count("!"),
        "body_urgency_count": sum(lower.count(w) for w in URGENCY_WORDS),
        "body_generic_greeting": int(any(g in lower for g in [
            "dear customer", "dear user", "dear valued", "valued customer", "dear account holder"])),
        "body_has_click_here": int("click here" in lower),
        "body_has_verify": int("verify" in lower),
        "body_has_credential": int(_has_any_word(lower, CREDENTIAL_WORDS)),
        "body_has_threat": int(_has_any_word(lower, THREAT_WORDS)),
        "body_has_prize": int(_has_any_word(lower, PRIZE_WORDS)),
        "body_has_time_pressure": int(_has_any_word(lower, TIME_PRESSURE)),
    }


def extract_url_features(body: str) -> dict:
    urls = _extract_urls(body)
    hosts = []
    for u in urls:
        try:
            host = urlsplit(u).hostname or ""
        except ValueError:
            host = ""
        if host:
            hosts.append(host)

    return {
        "url_count": len(urls),
        "url_unique_domains": len(set(hosts)),
        "url_has_ip": int(any(IPV4_RE.match(h) for h in hosts)),
        "url_has_at": int(any("@" in u for u in urls)),
        "url_has_http": int(any(u.startswith("http://") for u in urls)),
        "url_max_dots": max((h.count(".") for h in hosts), default=0),
        "url_max_domain_len": max((len(h) for h in hosts), default=0),
        "url_has_lookalike": int(any(_lookalike(h) for h in hosts)),
        "url_has_suspicious_tld": int(any(h.rsplit(".", 1)[-1] in SUSPICIOUS_TLDS for h in hosts)),
        "url_has_suspicious_path": int(any(any(k in u.lower() for k in SUSPICIOUS_PATH) for u in urls)),
    }


def extract_features(email: dict) -> dict:
    """合并三类特征为一个固定键集合的字典。"""
    feats: dict = {}
    feats.update(extract_sender_features(email.get("sender", "")))
    feats.update(extract_body_features(email.get("body", "")))
    feats.update(extract_url_features(email.get("body", "")))
    return feats


# 固定特征顺序（用空邮件推导，键的插入顺序稳定）
FEATURE_NAMES = list(extract_features({"sender": "", "body": "", "subject": ""}).keys())


# 可解释信号：特征名 -> 人类可读描述（仅在值为正时触发）
SIGNALS = [
    ("snd_is_free_email", "发件人使用免费邮箱"),
    ("snd_lookalike_brand", "发件人域名仿冒知名品牌"),
    ("snd_display_brand_mismatch", "发件人显示名与邮箱域名不符"),
    ("body_urgency_count", "正文包含紧迫性措辞"),
    ("body_exclamation_count", "正文使用大量感叹号"),
    ("body_generic_greeting", "使用泛化称呼（Dear Customer 等）"),
    ("body_has_click_here", "包含 Click here 诱导点击"),
    ("body_has_verify", "要求验证账户"),
    ("body_has_credential", "索取密码 / 凭证"),
    ("body_has_threat", "包含威胁（封号 / 停用）"),
    ("body_has_prize", "声称中奖 / 赠品"),
    ("body_has_time_pressure", "制造时间压力"),
    ("url_has_ip", "URL 使用 IP 地址"),
    ("url_has_at", "URL 包含 @ 符号"),
    ("url_has_http", "使用不加密的 http 链接"),
    ("url_has_lookalike", "URL 域名仿冒"),
    ("url_has_suspicious_tld", "URL 使用可疑顶级域名"),
    ("url_has_suspicious_path", "URL 路径诱导验证 / 登录"),
]


def explain_signals(feats: dict) -> list[str]:
    """返回被触发的可疑信号的可读说明。"""
    out = []
    for name, desc in SIGNALS:
        val = feats.get(name, 0)
        if isinstance(val, (int, float)) and val > 0:
            if name in ("body_urgency_count", "body_exclamation_count"):
                out.append(f"{desc}（{val} 次）")
            else:
                out.append(desc)
    return out


def email_text(email: dict) -> str:
    """拼接主题 + 正文，作为 TF-IDF 的输入文本。"""
    subject = str(email.get("subject", "") or "").strip()
    body = str(email.get("body", "") or "").strip()
    return f"{subject} {body}".strip()


def make_vectorizer(max_features: int = 5000) -> TfidfVectorizer:
    """TF-IDF 向量化器（1-2 gram，截断到 max_features 个词）。"""
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        lowercase=True,
        strip_accents="unicode",
    )


def stack_features(X_h: pd.DataFrame, texts: list[str], vectorizer: TfidfVectorizer | None = None, max_features: int = 5000):
    """把预计算的手写特征 X_h 与 TF-IDF(texts) 拼接成稀疏矩阵。

    vectorizer 为 None 时用 texts 拟合；否则只 transform。返回 (X, feature_names, vectorizer)。
    与 build_full_features 的区别：接受已算好的 X_h，避免重复提取手写特征。
    """
    if vectorizer is None:
        vectorizer = make_vectorizer(max_features=max_features)
        X_t = vectorizer.fit_transform(texts)
    else:
        X_t = vectorizer.transform(texts)

    tfidf_names = ["tfidf_" + w for w in vectorizer.get_feature_names_out()]
    X = hstack([csr_matrix(X_h.values), X_t]).tocsr()
    return X, list(X_h.columns) + tfidf_names, vectorizer


def build_full_features(records: list[dict], vectorizer: TfidfVectorizer | None = None, max_features: int = 5000):
    """手写特征 + TF-IDF 拼接成稀疏特征矩阵（单次调用，训练/预测共用）。

    vectorizer 为 None 时用 records 拟合（训练阶段）；否则只 transform（预测/测试阶段）。
    返回 (X, feature_names, vectorizer)。
    """
    X_h = pd.DataFrame([extract_features(r) for r in records])
    texts = [email_text(r) for r in records]
    return stack_features(X_h, texts, vectorizer, max_features)
