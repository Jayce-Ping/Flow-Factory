# src/flow_factory/utils/logger.py
import os
import logging
import torch

def get_rank():
    """Get process rank for distributed training."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', 0)))

def setup_logger(name: str = None, level: int = logging.INFO, rank_zero_only: bool = False):
    """
    Setup logger with rank information.
    
    Args:
        name: Logger name
        level: Logging level
        rank_zero_only: If True, only rank 0 will output logs
    """
    rank = get_rank()
    
    # Silence non-zero ranks if requested
    if rank_zero_only and rank != 0:
        logger = logging.getLogger(name)
        logger.setLevel(logging.CRITICAL + 1)  # Effectively disable
        return logger
    
    formatter = logging.Formatter(
        f'[%(asctime)s] [Rank {rank}] [%(levelname)s] [%(name)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.propagate = False
    
    return logger