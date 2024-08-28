# Copyright (c) 2024 Ian Hill
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

import jax
import jax.random as rand

import numpyro as npyro
from numpyro.handlers import seed, trace, condition


class ReliabilityModel():
    name: str
    test_spm: Callable
    life_spm: Callable
    test_measurements: list[str]
    measurements_per_chp: dict[int]
    life_predictors: list[str]
    life_conds: dict[str: float]
    hyls: list[str]
    hyl_beliefs: dict[dict[str: float]]

    def __init__(self, name, test_model, lifespan_model, param_vals, hyl_sites, prior, ltnt_sites, ltnt_subsample_sites,
                 meas_sites, meas_counts, pred_sites, pred_conds):
        self.name = name
        self.test_spm = test_model
        self.life_spm = lifespan_model

        self.param_vals = param_vals

        self.hyls = hyl_sites
        self.hyl_beliefs = prior

        self.ltnt_vals = ltnt_sites
        self._ltnt_subsamples = ltnt_subsample_sites

        self.test_measurements = meas_sites
        self.meas_per_chp = meas_counts

        self.life_predictors = pred_sites
        self.life_conds = pred_conds

    def validate_model(self):
        pass

    def sample_predictor_from_beliefs(self, rng_key: rand.key, predictor: str, num_samples: int = 1,
                                      lots_per_sample: int = 1, chps_per_sample: int = 1, field_use_conds = None):
        dims = {'lot': lots_per_sample, 'chp': chps_per_sample} | self.meas_per_chp
        conds = field_use_conds if field_use_conds is not None else self.life_conds[predictor]

        def sampler(rng):
            seeded = seed(self.life_spm, rng)
            tr = trace(seeded).get_trace(dims, conds, self.hyl_beliefs, self.param_vals)
            return tr[predictor]['value']

        return jax.vmap(sampler, axis_size=num_samples)(rand.split(rng_key, num_samples))

    def sample_predictor_from_hyl_vals(self, rng_key: rand.key, predictor: str, hyl_vals: dict[float],
                                       num_samples: int = 1, lots_per_sample: int = 1, chps_per_sample: int = 1):
        dims = {'lot': lots_per_sample, 'chp': chps_per_sample} | self.meas_per_chp

        def sampler(rng, vals):
            conditioned = condition(self.life_spm, data=vals)
            seeded = seed(conditioned, rng)
            tr = trace(seeded).get_trace(dims, self.life_conds, self.hyl_beliefs, self.param_vals)
            return tr[predictor]['value']

        return jax.vmap(sampler, axis_size=num_samples)(rand.split(rng_key, num_samples), hyl_vals)

    def logp_of_hyl_vals(self):
        pass

    def logp_of_latent_space_sample(self):
        pass
