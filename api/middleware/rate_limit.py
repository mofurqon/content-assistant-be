from config.settings import RATE_LIMIT_ENABLED

if RATE_LIMIT_ENABLED:
    from slowapi import Limiter
    from slowapi.util import get_remote_address
    _limiter: "Limiter | None" = Limiter(key_func=get_remote_address)
else:
    _limiter = None


def get_limiter() -> "Limiter | None":
    return _limiter


def rate_limit(limit_string: str):
    """
    Apply a slowapi rate limit on Railway; pass-through on local.
    Usage: @rate_limit("3/minute")
    """
    def decorator(func):
        if RATE_LIMIT_ENABLED and _limiter is not None:
            return _limiter.limit(limit_string)(func)
        return func
    return decorator
