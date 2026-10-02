"""合成演示数据生成器（增强版）。

生成更真实、更多样的带标签合成邮件（正常 / 钓鱼），覆盖多种真实钓鱼场景
（账户停用、假发票、包裹派送、密码过期、异常登录、中奖、退税、订阅续费、勒索、彩票），
以及多种发件人伪造与 URL 欺骗手法，用于跑通和演示「发件人/正文/URL」三类特征。

每条邮件是一个字典：
    sender   -- From 字段，形如 "PayPal Security <support@paypal-verify.com>"
    subject  -- 邮件主题
    body     -- 邮件正文（纯文本，含链接）
    label    -- 1 = 钓鱼, 0 = 正常

随机性由 seed 控制，结果可复现。
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

import config

# 品牌 -> 官方域名
BRANDS = {
    "PayPal": "paypal.com",
    "Apple": "apple.com",
    "Microsoft": "microsoft.com",
    "Amazon": "amazon.com",
    "Netflix": "netflix.com",
    "Google": "google.com",
    "Chase": "chase.com",
    "Bank of America": "bankofamerica.com",
    "Wells Fargo": "wellsfargo.com",
    "eBay": "ebay.com",
    "Dropbox": "dropbox.com",
    "LinkedIn": "linkedin.com",
}

COURIERS = {"DHL": "dhl.com", "FedEx": "fedex.com", "UPS": "ups.com", "USPS": "usps.com"}

FREE_EMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com",
    "live.com", "mail.ru", "protonmail.com", "qq.com", "163.com",
}

SUSPICIOUS_TLDS = {
    "xyz", "top", "club", "biz", "info", "click", "link", "tk", "ml",
    "ga", "cf", "gq", "work", "zip", "online", "site", "icu", "rest", "support",
}

FIRST_NAMES = ["Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Sam", "Chris", "Jamie", "Robin"]


def _lookalike_domain(brand: str, rng: random.Random) -> str:
    """把品牌名改造成钓鱼常见的仿冒域名，如 paypal -> paypal-secure.xyz。

    保留品牌子串，便于「仿冒品牌」特征稳定识别；偶尔做字符替换（l->1, o->0）。
    """
    name = brand.lower().replace(" ", "")
    mutated = name.replace("l", "1", 1).replace("o", "0", 1)
    style = rng.choice([
        f"{name}-secure.{rng.choice(['com', 'net', 'xyz', 'top'])}",
        f"{name}-verify.{rng.choice(['com', 'net', 'club', 'top'])}",
        f"{name}-login.{rng.choice(['com', 'xyz', 'biz', 'top'])}",
        f"{name}-account.{rng.choice(['com', 'info', 'xyz'])}",
        f"{name}.com-{rng.choice(['secure', 'verify', 'login', 'update'])}.{rng.choice(['xyz', 'top', 'com', 'net'])}",
        f"secure-{name}-{rng.choice(['verify', 'login', 'signin'])}.{rng.choice(['com', 'net', 'top'])}",
        f"{mutated}-{rng.choice(['secure', 'verify', 'login'])}.{rng.choice(['com', 'net', 'xyz'])}",
    ])
    return style


def _spoofed_sender(brand: str, rng: random.Random, lookalike: str) -> str:
    """构造伪造发件人：仿冒域名 / 免费邮箱带品牌名 / 显示名伪装。"""
    name = rng.choice([
        f"{brand} Security", f"{brand} Support", f"{brand} Alert", f"{brand} Billing",
        f"{brand} Account Team", "Security Center", f"{brand} Verification", "Account Services",
    ])
    style = rng.random()
    if style < 0.4:
        local = rng.choice(["security", "support", "noreply", "alert", "verify", "billing", "account", "no-reply"])
        return f"{name} <{local}@{lookalike}>"
    if style < 0.7:
        domain = rng.choice(list(FREE_EMAIL_DOMAINS))
        slug = brand.lower().replace(" ", "")
        local = f"{slug}.{rng.choice(['security', 'support', 'alert', 'verify', 'service', 'team'])}"
        if rng.random() < 0.5:
            local += str(rng.randint(1, 999))
        return f"{name} <{local}@{domain}>"
    domain = rng.choice(["mail-secure.com", "account-verify.net", "security-alerts.info", "notification-center.com", lookalike])
    local = rng.choice(["security", "alert", "noreply", "verify", "support"])
    return f"{name} <{local}{rng.randint(10, 999)}@{domain}>"


def _phishing_url(rng: random.Random, lookalike: str) -> str:
    """构造钓鱼 URL，轮换多种欺骗手法：仿冒域名/IP/子域名/@/短链/可疑 TLD。"""
    scheme = rng.choice(["http", "http", "https"])
    path = rng.choice(["verify", "login", "account", "secure", "update", "confirm", "signin", "unlock", "recover", "validate"])
    trick = rng.random()
    if trick < 0.34:
        return f"{scheme}://{lookalike}/{path}"
    if trick < 0.52:
        ip = f"{rng.randint(1, 254)}.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(0, 255)}"
        return f"{scheme}://{ip}/{path}"
    if trick < 0.68:
        official = rng.choice(list(BRANDS.values()))
        return f"{scheme}://{official}.{lookalike}/{path}"
    if trick < 0.80:
        official = rng.choice(list(BRANDS.values()))
        return f"{scheme}://{lookalike}@{official}/{path}"
    if trick < 0.90:
        code = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(7))
        return f"http://bit.ly/{code}"
    base = lookalike.rsplit(".", 1)[0]
    return f"{scheme}://{base}.{rng.choice(list(SUSPICIOUS_TLDS))}/{path}"


# ---------------------------------------------------------------------------
# 钓鱼场景：每个函数返回完整的 (sender, subject, body)
# ---------------------------------------------------------------------------

def _phish_account(brand, rng, lookalike):
    url = _phishing_url(rng, lookalike)
    sender = _spoofed_sender(brand, rng, lookalike)
    subjects = [
        f"URGENT: Your {brand} account has been suspended",
        f"Action required: {brand} account temporarily limited",
        f"Your {brand} account has been locked",
    ]
    bodies = [
        f"Dear Customer,\n\nWe have detected unusual activity on your {brand} account. "
        f"For your protection, your account has been temporarily suspended. "
        f"To restore access you MUST verify your identity within 24 hours or your account will be permanently closed.\n\n"
        f"Click here to verify now: {url}\n\nFailure to act immediately will result in account termination.\n\n{brand} Security Team",
        f"Dear Valued User,\n\nYour {brand} account requires immediate verification due to recent policy updates. "
        f"Your account has been restricted until you confirm your information.\n\nVerify your account here: {url}\n\n"
        f"Warning: failure to verify within 48 hours will result in permanent suspension of your account!!!",
    ]
    return {"sender": sender, "subject": rng.choice(subjects), "body": rng.choice(bodies), "label": 1}


def _phish_invoice(brand, rng, lookalike):
    url = _phishing_url(rng, lookalike)
    sender = _spoofed_sender(brand, rng, lookalike)
    amount = rng.choice(["$149.99", "$299.00", "$89.95", "$499.99"])
    invoice = rng.randint(10000, 99999)
    subjects = [f"Invoice #{invoice} from {brand}", f"Your payment receipt {invoice}", f"Order confirmation #{invoice}"]
    bodies = [
        f"Dear Customer,\n\nThank you for your purchase. Your order #{invoice} for {amount} has been confirmed. "
        f"If you did NOT authorize this charge, you must dispute it within 24 hours to avoid being billed.\n\n"
        f"Click here to cancel or dispute: {url}\n\n{brand} Billing Department",
        f"Dear Customer,\n\nYour account was charged {amount}. Attached is your invoice #{invoice}. "
        f"To view the invoice or report an unauthorized charge, click here: {url}\n\n"
        f"This charge will post to your account immediately if no action is taken.",
    ]
    return {"sender": sender, "subject": rng.choice(subjects), "body": rng.choice(bodies), "label": 1}


def _phish_delivery(brand, rng, lookalike):
    courier, official = rng.choice(list(COURIERS.items()))
    lookalike = _lookalike_domain(courier, rng)
    url = _phishing_url(rng, lookalike)
    local = rng.choice(["noreply", "delivery", "tracking", "notification", "parcel"])
    sender = f"{courier} Notification <{local}{rng.randint(1, 99)}@{lookalike}>"
    parcel = rng.choice(["#GH" + str(rng.randint(100000, 999999)), "#A" + str(rng.randint(1000000, 9999999))])
    subjects = [f"Unable to deliver parcel {parcel}", f"Delivery failed for {parcel}", f"Action needed: {parcel} held at depot"]
    bodies = [
        f"Dear Customer,\n\nOur courier was unable to deliver your parcel {parcel} due to an incorrect address. "
        f"A small redelivery fee is required to complete delivery.\n\n"
        f"Confirm your address and pay the fee here: {url}\n\n"
        f"Your parcel will be returned to sender within 48 hours if no action is taken.\n\n{courier} Team",
        f"Dear Customer,\n\nYour parcel {parcel} is being held at our local depot. "
        f"To schedule redelivery you must confirm your shipping details within 24 hours.\n\n"
        f"Click here to confirm: {url}\n\nFailure to respond will result in the parcel being discarded.",
    ]
    return {"sender": sender, "subject": rng.choice(subjects), "body": rng.choice(bodies), "label": 1}


def _phish_password(brand, rng, lookalike):
    url = _phishing_url(rng, lookalike)
    sender = _spoofed_sender(brand, rng, lookalike)
    subjects = [f"Your {brand} password expires today", f"Password reset required for {brand}", f"Security update: change your password"]
    bodies = [
        f"Dear User,\n\nYour {brand} password will expire in 24 hours. To keep your current password, "
        f"click below and sign in with your credentials.\n\nKeep my password: {url}\n\n"
        f"If you do not act, your account will be disabled for security reasons.\n\n{brand} IT Service Desk",
        f"Dear User,\n\nWe are upgrading our security systems. All accounts must update their password "
        f"to avoid service interruption. Enter your current password here to confirm your identity:\n\n{url}\n\n"
        f"Warning: accounts not updated within 48 hours will be locked.",
    ]
    return {"sender": sender, "subject": rng.choice(subjects), "body": rng.choice(bodies), "label": 1}


def _phish_alert(brand, rng, lookalike):
    url = _phishing_url(rng, lookalike)
    sender = _spoofed_sender(brand, rng, lookalike)
    subjects = [f"New sign-in from an unknown device", f"Suspicious login attempt on your {brand} account", f"Security alert: someone accessed your account"]
    bodies = [
        f"Dear Customer,\n\nWe detected a new sign-in to your {brand} account from an unknown device and location. "
        f"If this was not you, your account may be compromised.\n\n"
        f"Review recent activity and secure your account here: {url}\n\n"
        f"Your account will be temporarily blocked if you do not confirm within 24 hours.",
        f"Security Alert,\n\nA login to your {brand} account was blocked from an unrecognized browser. "
        f"To regain access, verify your identity immediately: {url}\n\n"
        f"Failure to verify may result in permanent account restriction.",
    ]
    return {"sender": sender, "subject": rng.choice(subjects), "body": rng.choice(bodies), "label": 1}


def _phish_prize(brand, rng, lookalike):
    url = _phishing_url(rng, lookalike)
    domain = rng.choice(list(FREE_EMAIL_DOMAINS))
    local = f"{brand.lower().replace(' ', '')}.rewards{rng.randint(1, 999)}"
    sender = f"{brand} Rewards <{local}@{domain}>"
    prize = rng.choice(["$500 gift card", "a free iPhone", "$1,000 cash", "a $250 voucher"])
    subjects = [f"Congratulations! You have been selected", f"Claim your {prize}", "You are today's lucky winner!"]
    bodies = [
        f"CONGRATULATIONS!\n\nYou have won {prize}! You were automatically entered into our promotion and selected as today's winner.\n\n"
        f"Claim your reward now before it expires: {url}\n\n"
        f"Act quickly! This offer is valid for only 24 hours. Click the link to claim your prize!!!",
        f"Dear Customer,\n\nYou have been selected to receive {prize}. "
        f"To claim your reward, confirm your delivery details within 24 hours.\n\nClaim here: {url}\n\n"
        f"This is a limited-time offer and will expire soon!!!",
    ]
    return {"sender": sender, "subject": rng.choice(subjects), "body": rng.choice(bodies), "label": 1}


def _phish_tax(brand, rng, lookalike):
    lookalike = "irs-refund.com"
    url = _phishing_url(rng, lookalike)
    domain = rng.choice(list(FREE_EMAIL_DOMAINS))
    sender = f"IRS Refund Department <refund@{domain}>"
    amount = rng.choice(["$1,247.30", "$892.15", "$2,110.50", "$643.20"])
    subjects = ["Your tax refund is ready", f"IRS notice: refund of {amount} pending", "Important: tax refund confirmation"]
    bodies = [
        f"Dear Taxpayer,\n\nAfter our final review we determined you are eligible for a tax refund of {amount}. "
        f"To receive your refund you must confirm your bank account details within 24 hours.\n\n"
        f"Confirm your details here: {url}\n\n"
        f"Failure to claim within the deadline will forfeit your refund.\n\nInternal Revenue Service",
        f"Dear Taxpayer,\n\nOur records indicate an unclaimed refund of {amount}. "
        f"Please verify your identity and banking information to process your payment.\n\nVerify here: {url}\n\n"
        f"Your refund will be held for 48 hours before being returned to the treasury.",
    ]
    return {"sender": sender, "subject": rng.choice(subjects), "body": rng.choice(bodies), "label": 1}


def _phish_subscription(brand, rng, lookalike):
    url = _phishing_url(rng, lookalike)
    sender = _spoofed_sender(brand, rng, lookalike)
    subjects = [f"Your {brand} subscription has been paused", f"Payment method failed for {brand}", f"Update your billing information"]
    bodies = [
        f"Dear Customer,\n\nWe could not process your latest payment for {brand}. "
        f"Your subscription has been paused and will be cancelled in 24 hours if not resolved.\n\n"
        f"Update your billing information here: {url}\n\n"
        f"To avoid service interruption, please act immediately.",
        f"Dear Customer,\n\nYour payment method on file has expired. "
        f"Update your payment details to keep your {brand} account active.\n\nUpdate now: {url}\n\n"
        f"Accounts not updated within 48 hours will be deactivated.",
    ]
    return {"sender": sender, "subject": rng.choice(subjects), "body": rng.choice(bodies), "label": 1}


def _phish_extortion(brand, rng, lookalike):
    domain = rng.choice(list(FREE_EMAIL_DOMAINS))
    local = rng.choice(["hacker", "anonymous", "darknet", "security", "unknown"])
    sender = f"{local}{rng.randint(10, 999)}@{domain}"
    amount = rng.choice(["$800", "$1,500", "$2,000"])
    wallet = "bc1" + "".join(rng.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(30))
    subjects = ["I hacked your webcam", "I know what you did", "Read this carefully - your privacy is at risk"]
    bodies = [
        f"I have bad news for you.\n\nA few months ago I gained access to your device and I have been watching you. "
        f"I recorded your screen, your webcam, and your browsing history. "
        f"If you do not want me to send this material to all your contacts, transfer {amount} in Bitcoin to the wallet below within 48 hours.\n\n"
        f"{wallet}\n\nDo not contact the police. This message will self-destruct after the deadline.",
        f"You do not know me but I know everything about you.\n\n"
        f"I have full control of your device and collected sensitive information about you. "
        f"Pay {amount} to the wallet below within 48 hours or your secrets will be published.\n\n{wallet}\n\n"
        f"Think carefully. Your reputation is at stake.",
    ]
    return {"sender": sender, "subject": rng.choice(subjects), "body": rng.choice(bodies), "label": 1}


def _phish_lottery(brand, rng, lookalike):
    domain = rng.choice(list(FREE_EMAIL_DOMAINS))
    local = rng.choice(["claims", "agent", "lotterydept", "prizeoffice"])
    sender = f"Lottery Department <{local}.{rng.randint(10, 999)}@{domain}>"
    url = _phishing_url(rng, "lottery-claim.com")
    amount = rng.choice(["$5,000,000", "$1,250,000", "$750,000"])
    subjects = ["You are a lottery winner!", "Unclaimed prize notification", f"Congratulations on winning {amount}"]
    bodies = [
        f"Dear Winner,\n\nYour email address was selected in our international lottery draw. "
        f"You have won {amount}. To claim your prize, contact our claims agent and provide your bank account details.\n\n"
        f"Claim your winnings here: {url}\n\nThis offer is time-sensitive and expires in 48 hours.",
        f"Dear Winner,\n\nCongratulations! You have been awarded {amount} in our annual sweepstakes. "
        f"Please confirm your identity and banking information to receive your funds.\n\nConfirm here: {url}",
    ]
    return {"sender": sender, "subject": rng.choice(subjects), "body": rng.choice(bodies), "label": 1}


PHISHING_SCENARIOS = [
    _phish_account, _phish_invoice, _phish_delivery, _phish_password, _phish_alert,
    _phish_prize, _phish_tax, _phish_subscription, _phish_extortion, _phish_lottery,
]


def _gen_phishing(rng: random.Random) -> dict:
    brand = rng.choice(list(BRANDS.keys()))
    lookalike = _lookalike_domain(brand, rng)
    scenario = rng.choice(PHISHING_SCENARIOS)
    return scenario(brand, rng, lookalike)


# ---------------------------------------------------------------------------
# 正常邮件场景
# ---------------------------------------------------------------------------

def _gen_legitimate(rng: random.Random) -> dict:
    brand = rng.choice(list(BRANDS.keys()))
    domain = BRANDS[brand]
    name = rng.choice(FIRST_NAMES)
    greeting = rng.choice([f"Hi {name},", f"Hello {name},", f"Dear {name},"])
    prefix = rng.choice(["noreply", "no-reply", "support", "service", "updates", "info", "help", "billing"])
    sender = f"{brand} <{prefix}@{domain}>"
    account = f"https://www.{domain}/account"
    orders = f"https://www.{domain}/orders"
    kind = rng.random()

    if kind < 0.16:  # 订单确认
        subjects = [f"Your {brand} order has shipped", "Order confirmation", "Thank you for your order"]
        bodies = [
            f"{greeting}\n\nThank you for your purchase. Your order has been confirmed and will ship within 2-3 business days. "
            f"Track your order at {orders}\n\nYou can manage your account at {account}.\n\nBest regards,\n{brand}",
            f"{greeting}\n\nYour order is confirmed. We will notify you when it ships. "
            f"View order details here: {orders}\n\nQuestions? Contact our support team.\n\n{brand}",
        ]
    elif kind < 0.32:  # 物流通知
        subjects = [f"A package is on its way from {brand}", "Shipping update", "Your delivery is scheduled"]
        bodies = [
            f"{greeting}\n\nGood news! Your package is on its way and is scheduled for delivery this week. "
            f"Track your shipment at {orders}\n\n{brand} Delivery",
            f"{greeting}\n\nYour shipment is confirmed. You can view the estimated delivery date in your account: {account}\n\n"
            f"Thanks for shopping with us.\n\n{brand}",
        ]
    elif kind < 0.48:  # 简报/更新
        subjects = [f"Weekly newsletter from {brand}", f"What's new at {brand}", "Your monthly update"]
        bodies = [
            f"{greeting}\n\nHere is your weekly update with the latest news and offers. "
            f"Read the full newsletter here: https://www.{domain}/newsletter\n\n"
            f"You can unsubscribe at any time from your account settings: {account}\n\n{brand}",
            f"{greeting}\n\nWe have some updates you might like. View them at https://www.{domain}/updates\n\n"
            f"Manage your email preferences at {account}.\n\n{brand}",
        ]
    elif kind < 0.62:  # 收据
        subjects = [f"Receipt for your recent purchase", "Your payment receipt", "Transaction confirmation"]
        bodies = [
            f"{greeting}\n\nThis email confirms your recent transaction. A receipt is available for your records: "
            f"{orders}\n\nIf you have questions about this transaction, contact our billing team.\n\n{brand} Billing",
        ]
    elif kind < 0.76:  # 对账单
        subjects = [f"Your monthly statement is ready", "Account statement available", "Your recent account activity"]
        bodies = [
            f"{greeting}\n\nYour latest monthly statement is now available for review in your account center: {account}\n\n"
            f"If you notice any unfamiliar activity, please contact us.\n\nRegards,\n{brand}",
        ]
    elif kind < 0.88:  # 安全提醒（正当，信息性，无验证链接）
        subjects = [f"New sign-in from your device", "Security notice: new device sign-in"]
        bodies = [
            f"{greeting}\n\nWe noticed a new sign-in to your account from a device we do not recognize. "
            f"If this was you, no action is needed.\n\nYou can review recent sign-ins at {account}.\n\n{brand} Security",
        ]
    else:  # 密码重置请求（用户主动发起，官方域名）
        subjects = ["Password reset request", "Reset your password"]
        bodies = [
            f"{greeting}\n\nWe received a request to reset your password. If you made this request, "
            f"click the link below to set a new password:\n\nhttps://www.{domain}/reset-password\n\n"
            f"If you did not request this, you can safely ignore this email.\n\n{brand}",
        ]

    return {
        "sender": sender,
        "subject": rng.choice(subjects),
        "body": rng.choice(bodies),
        "label": 0,
    }


def generate_dataset(n_normal: int = 250, n_phishing: int = 250, seed: int = config.SEED) -> list[dict]:
    """生成 n_normal 条正常 + n_phishing 条钓鱼邮件，打乱顺序后返回。"""
    rng = random.Random(seed)
    records = [_gen_legitimate(rng) for _ in range(n_normal)]
    records += [_gen_phishing(rng) for _ in range(n_phishing)]
    rng.shuffle(records)
    return records


def save_dataset(records: list[dict], path: Path | str = config.DATA_PATH) -> Path:
    """把邮件记录写成 CSV，返回写入路径。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(path, index=False, encoding="utf-8")
    print(f"已生成 {len(records)} 条邮件，保存到 {path}")
    return path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="生成合成演示数据")
    parser.add_argument("--n-normal", type=int, default=250)
    parser.add_argument("--n-phishing", type=int, default=250)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--out", type=str, default=str(config.DATA_PATH))
    args = parser.parse_args()

    records = generate_dataset(args.n_normal, args.n_phishing, args.seed)
    save_dataset(records, args.out)
