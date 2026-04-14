#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
说明: BEV-TextCLIP 多模态语义分割模型包
日期: 2026年1月22日
"""

from .configs import BEVTextCLIPConfig, get_config
from .models import *
from .dataloaders import *
from .utils import *
from .visualization import SegmentationVisualizer

__version__ = "1.0.0"
