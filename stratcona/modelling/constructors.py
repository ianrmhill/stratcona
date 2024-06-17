# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import pymc

__all__ = ['get_inference_version']


def get_inference_version(model: pymc.Model, observations: dict):
    return pymc.model.transform.conditioning.observe(model, observations)


def add_composite_variable(model_builder, var_name, prm_dists_and_priors, composition='sum'):
    model_builder.add_latent_variable(f"{var_name}_mean", pymc.Normal, {'mu': 0.7, 'sigma': 0.2})
    model_builder.add_latent_variable(f"{var_name}_dev", pymc.Normal, {'mu': 0.7, 'sigma': 0.2})

    def composition():
        pass
