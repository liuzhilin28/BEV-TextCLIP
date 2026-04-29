#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
说明: BEV-TextCLIP 多模态语义分割模型包
日期: 2026年1月22日
"""

import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HF_HOME = os.environ.setdefault(
    'HF_HOME',
    os.path.join(_REPO_ROOT, '.cache', 'huggingface'),
)
os.environ.setdefault('HF_HUB_CACHE', os.path.join(_HF_HOME, 'hub'))
os.environ.setdefault('HF_XET_CACHE', os.path.join(_HF_HOME, 'xet'))
os.environ.setdefault('TRANSFORMERS_CACHE', os.path.join(_HF_HOME, 'transformers'))
os.environ.setdefault('MPLCONFIGDIR', os.path.join(_REPO_ROOT, '.cache', 'matplotlib'))

from .configs import BEVTextCLIPConfig, get_config

__version__ = "1.0.0"

__all__ = ["BEVTextCLIPConfig", "get_config", "__version__"]
