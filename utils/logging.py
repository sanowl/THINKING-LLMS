import logging
from rich.logging import RichHandler
from typing import Optional

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a configured logger instance."""
    logger = logging.getLogger(name or "tpo")
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = RichHandler(rich_tracebacks=True)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    
    return logger