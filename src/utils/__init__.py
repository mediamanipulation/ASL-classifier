"""Utility functions"""
from .logging import setup_logging
from .metrics import evaluate_model

__all__ = ['setup_logging', 'evaluate_model']
