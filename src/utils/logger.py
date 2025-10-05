"""
Logging utilities for the RAG system.
Provides structured logging with different levels and outputs.
"""

import sys
import logging
from pathlib import Path
from typing import Optional
from loguru import logger
from .config import config


class Logger:
    """
    Custom logger wrapper using loguru.
    Provides structured logging with file rotation and different levels.
    """
    
    def __init__(
        self,
        name: str = "compliance_rag",
        level: str = "INFO",
        log_file: Optional[str] = None
    ):
        """
        Initialize logger.
        
        Args:
            name: Logger name
            level: Log level (DEBUG, INFO, WARNING, ERROR)
            log_file: Optional log file path
        """
        self.name = name
        self.level = level
        
        # Remove default logger
        logger.remove()
        
        # Add console handler with custom format
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
            level=level,
            colorize=True
        )
        
        # Add file handler if specified
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.add(
                log_file,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
                level=level,
                rotation="10 MB",  # Rotate when file reaches 10MB
                retention="30 days",  # Keep logs for 30 days
                compression="zip"  # Compress rotated logs
            )
    
    def get_logger(self):
        """Get the logger instance"""
        return logger


# Global logger instance
app_logger = Logger(
    name="compliance_rag",
    level=config.get("logging.level", "INFO"),
    log_file=config.get("logging.file", "./logs/app.log")
).get_logger()

