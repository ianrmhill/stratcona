# Copyright (c) 2024 Ian Hill
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as rand

from numpyro.handlers import seed, trace, condition


class ReliabilityRequirement():
    metric: Callable
    quantile: float
    target_lifespan: float

    def __init__(self, metric, quantile, target_lifespan):
        self.type = metric
        self.quantile = quantile
        self.target_lifespan = target_lifespan


class ReliabilityTest():
    config: dict[dict[int]]
    conditions: dict[dict[float]]

    def __init__(self, config, conditions):
        self.config = config
        self.conditions = conditions


class ReliabilityModel():
    name: str
    test_spm: Callable
    test_measurements: list[str]
    measurements_per_chp: dict[int]
    life_predictors: list[str]
    life_conds: dict[str: float]
    hyls: dict[str]
    hyl_beliefs: dict[dict[str: float]]
    hyl_info: dict[dict]

    def __init__(self, name, test_model, param_vals, hyl_sites, prior, hyl_info, ltnt_sites,
                 ltnt_subsample_sites, meas_sites, meas_counts, pred_sites, pred_conds):
        self.name = name
        self.test_spm = test_model

        self.param_vals = param_vals

        self.hyls = hyl_sites
        self.hyl_beliefs = prior
        self.hyl_info = hyl_info

        self.ltnts = ltnt_sites
        self._ltnt_subsamples = ltnt_subsample_sites

        self.test_measurements = meas_sites
        self.meas_per_chp = meas_counts

        self.life_predictors = pred_sites
        self.life_conds = pred_conds

    def validate_model(self):
        # TODO
        pass

    def sample(self, rng_key: rand.key, test: ReliabilityTest, num_samples: tuple = (), keep_sites: list = None,
                   conditionals: dict = None, full_trace=False, alt_priors=None):
        # Convert any keep_sites that need more complex node names
        if keep_sites is not None:
            sites = []
            for i, site in enumerate(keep_sites):
                if site in self.test_measurements:
                    for tst in test.config:
                        sites.append(f'{tst}_{site}')
                else:
                    sites.append(site)
        else:
            sites = None

        priors = self.hyl_beliefs if alt_priors is None else alt_priors

        def sampler(rng, set_vals):
            mdl = self.test_spm if set_vals is None else condition(self.test_spm, data=set_vals)
            seeded = seed(mdl, rng)
            tr = trace(seeded).get_trace(test.config, test.conditions, priors, self.param_vals)
            samples = {site: tr[site] if full_trace else tr[site]['value'] for site in tr}
            return dict((k, samples[k]) for k in sites) if sites is not None else samples

        wrapped = sampler
        size = 1
        for i, dim in enumerate(num_samples):
            size *= dim
            wrapped = jax.vmap(wrapped, 0, 0)
        keys = jnp.reshape(rand.split(rng_key, size), num_samples)

        return wrapped(keys, conditionals)

    def logp(self, rng_key: rand.key, test: ReliabilityTest, site_vals: dict, conditional: dict | None, dims: tuple = ()):

        def get_log_prob(rng, vals, cond):
            mdl = self.test_spm if cond is None else condition(self.test_spm, data=cond)
            seeded = seed(mdl, rng)
            tr = trace(seeded).get_trace(test.config, test.conditions, self.hyl_beliefs, self.param_vals)
            lp = 0
            for site in vals:
                lp += jnp.sum(tr[site]['fn'].log_prob(vals[site]))
            return lp

        wrapped = get_log_prob
        size = 1
        for dim in dims:
            size *= dim
            wrapped = jax.vmap(wrapped, 0, 0)

        keys = jnp.reshape(rand.split(rng_key, size), dims)
        return wrapped(keys, site_vals, conditional)
