# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import pymc

__all__ = ['get_inference_version']


def get_inference_version(model: pymc.Model, observations: dict):
    return pymc.model.transform.conditioning.observe(model, observations)
