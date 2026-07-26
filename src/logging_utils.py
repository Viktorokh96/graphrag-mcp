"""Единая настройка логирования для всех точек входа (MCP-сервер, CLI)."""

import logging
import os
import sys

#: Библиотечные логгеры, которые только шумят на INFO
NOISY_LOGGERS = ("httpx", "huggingface_hub", "sentence_transformers", "httpcore")

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def configure_logging(default_level: str = "INFO") -> None:
    """Настроить корневой логгер (уровень из LOG_LEVEL) и приглушить библиотеки."""
    level_name = os.environ.get("LOG_LEVEL", default_level).upper()
    logging.basicConfig(
        level=getattr(logging, level_name, logging.INFO),
        format=LOG_FORMAT,
        stream=sys.stderr,
    )
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
