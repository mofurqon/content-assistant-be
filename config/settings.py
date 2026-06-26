import os
from dotenv import load_dotenv

load_dotenv()


def require_env(name: str) -> str:
    """
    Return the value of a required environment variable.

    Raises a clear, actionable error if it is missing or empty, instead of
    a bare KeyError. Add the variable to your .env file (see .env.example).
    """
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            f"Set it in your .env file (see .env.example)."
        )
    return value
