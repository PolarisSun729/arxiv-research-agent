"""密码只使用 bcrypt 哈希；按 UTF-8 字节校验，避免超过 72 字节后静默截断。"""

from functools import lru_cache

import bcrypt


# bcrypt 成本因子：每次哈希迭代 2^12 轮。调高会让登录和注册更慢，也让离线暴力破解更难。
BCRYPT_ROUNDS = 12


def validate_password(password: str) -> str:
    """校验新密码的强度，合格则原样返回，否则抛出 ValueError。

    规则：至少 8 个字符；UTF-8 编码后不超过 72 字节（bcrypt 的输入上限）；
    同时包含大写字母、小写字母和数字。
    """
    if not 8 <= len(password) or len(password.encode("utf-8")) > 72:
        raise ValueError("密码至少 8 个字符，UTF-8 编码后最多 72 字节")
    if not all((any(c.isupper() for c in password), any(c.islower() for c in password), any(c.isdigit() for c in password))):
        raise ValueError("密码必须包含大写字母、小写字母和数字")
    return password


def hash_password(password: str) -> str:
    """校验强度后生成 bcrypt 哈希（自带随机盐），返回可直接入库的 ASCII 字符串。"""
    return bcrypt.hashpw(validate_password(password).encode("utf-8"), bcrypt.gensalt(rounds=BCRYPT_ROUNDS)).decode("ascii")


@lru_cache(maxsize=2)
def _dummy_hash(rounds: int) -> bytes:
    """用户不存在时用来比较的假哈希；按成本因子缓存，只在首次使用时计算一次。"""
    return bcrypt.hashpw(b"unused-account-comparison", bcrypt.gensalt(rounds=rounds))


def verify_password(password: str, hashed: str | None) -> bool:
    """校验明文密码与哈希是否匹配；hashed 为 None 表示用户不存在，此时始终返回 False。

    这里不检查密码复杂度：登录时只判断是否匹配，不应向调用方透露密码策略。
    """
    encoded = password.encode("utf-8")
    # 空密码或超过 bcrypt 上限的输入不可能是合法密码，直接拒绝。
    if not encoded or len(encoded) > 72:
        return False
    # 未知用户也执行相同成本的比较，避免仅凭快速响应判断用户名是否存在。
    candidate_hash = hashed.encode("ascii") if hashed else _dummy_hash(BCRYPT_ROUNDS)
    try:
        matched = bcrypt.checkpw(encoded, candidate_hash)
        # 与假哈希比较的结果一律不算匹配。
        return bool(hashed) and matched
    except (ValueError, TypeError):
        # 数据库中的哈希格式损坏时按校验失败处理，不抛出异常。
        return False
