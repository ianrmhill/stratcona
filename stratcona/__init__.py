# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

"""
Stratcona optimal experimental design and inference framework.
"""

__version__ = '0.0.5'

from . import assistants, engine, modelling
from .modelling import SPMBuilder, AnalysisManager, ReliabilityModel, ReliabilityRequirement, ReliabilityTest

__all__ = []
