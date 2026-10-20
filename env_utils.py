import os
from pathlib import Path

from dotenv import load_dotenv


def load_environment() -> None:
    root_dir = Path(__file__).resolve().parent
    load_dotenv(dotenv_path=root_dir / ".env", override=False)
    load_dotenv(override=False)


def require_env(name: str) -> str:
    load_environment()
    value = os.getenv(name)
    if value is None or value == "":
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            "Copy .env.example to .env and fill in the values."
        )
    return value
