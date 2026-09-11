"""密码只使用 bcrypt 哈希；按 UTF-8 字节校验，避免超过 72 字节后静默截断。"""

from functools import lru_cache

import bcrypt


BCRYPT_ROUNDS = 12


def validate_password(password: str) -> str:
    if not 8 <= len(password) or len(password.encode("utf-8")) > 72:
        raise ValueError("密码至少 8 个字符，UTF-8 编码后最多 72 字节")
    if not all((any(c.isupper() for c in password), any(c.islower() for c in password), any(c.isdigit() for c in password))):
        raise ValueError("密码必须包含大写字母、小写字母和数字")
    return password


def hash_password(password: str) -> str:
    return bcrypt.hashpw(validate_password(password).encode("utf-8"), bcrypt.gensalt(rounds=BCRYPT_ROUNDS)).decode("ascii")


@lru_cache(maxsize=2)
def _dummy_hash(rounds: int) -> bytes:
    return bcrypt.hashpw(b"unused-account-comparison", bcrypt.gensalt(rounds=rounds))


def verify_password(password: str, hashed: str | None) -> bool:
    encoded = password.encode("utf-8")
    if not encoded or len(encoded) > 72:
        return False
    # 未知用户也执行相同成本的比较，避免仅凭快速响应判断用户名是否存在。
    candidate_hash = hashed.encode("ascii") if hashed else _dummy_hash(BCRYPT_ROUNDS)
    try:
        matched = bcrypt.checkpw(encoded, candidate_hash)
        return bool(hashed) and matched
    except (ValueError, TypeError):
        return False
