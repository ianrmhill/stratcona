# Copyright (c) 2024 Ian Hill
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as rand

import numpyro as npyro
from numpyro.handlers import seed, trace, condition


class ReliabilityRequirement():
    type: str
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
    life_spm: Callable
    test_measurements: list[str]
    measurements_per_chp: dict[int]
    life_predictors: list[str]
    life_conds: dict[str: float]
    hyls: dict[str]
    hyl_beliefs: dict[dict[str: float]]
    hyl_info: dict[dict]

    def __init__(self, name, test_model, lifespan_model, param_vals, hyl_sites, prior, hyl_info, ltnt_sites,
                 ltnt_subsample_sites, meas_sites, meas_counts, pred_sites, pred_conds):
        self.name = name
        self.test_spm = test_model
        self.life_spm = lifespan_model

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
                                       num_samples: int = 1, chps_per_sample: int = 1, lots_per_sample: int = 1):
        dims = {'lot': lots_per_sample, 'chp': chps_per_sample} | self.meas_per_chp

        def sampler(rng, vals):
            conditioned = condition(self.life_spm, data=vals)
            seeded = seed(conditioned, rng)
            tr = trace(seeded).get_trace(dims, self.life_conds, self.hyl_beliefs, self.param_vals)
            return tr[predictor]['value']

        return jax.vmap(sampler, axis_size=num_samples)(rand.split(rng_key, num_samples), hyl_vals)

    def sample_measurements(self, rng_key: rand.key, test_config: dict = None, test_conditions: dict = None,
                            latent_vals: dict = None, priors: dict = None, num_samples: int = 1):
        dims = {}
        for exp in test_config:
            dims[exp] = test_config[exp] | self.meas_per_chp
        pri = priors if priors is not None else self.hyl_beliefs
        if latent_vals is not None:
            sample_mdl = condition(self.test_spm, data=latent_vals)
        else:
            sample_mdl = self.test_spm

        def sampler(rng):
            seeded = seed(sample_mdl, rng)
            tr = trace(seeded).get_trace(dims, test_conditions, pri, self.param_vals)
            measured = {}
            for exp in dims:
                measured[exp] = {}
                for meas in self.test_measurements:
                    measured[exp][meas] = tr[f'{exp}_{meas}']['value']
            return measured

        return jax.vmap(sampler, axis_size=num_samples)(rand.split(rng_key, num_samples))

    def sample(self, rng_key: rand.key, test: ReliabilityTest, num_samples: int = 1, keep_sites: list = None,
               hyl_vals: dict[float] = None, ltnt_vals: dict = None, meas_vals: dict[float] = None,
               alt_priors: dict[dict[float]] = None):
        pri = alt_priors if alt_priors is not None else self.hyl_beliefs

        # Now handle any conditioning where sample sites are fixed to values
        mdl = self.test_spm if hyl_vals is None else condition(self.test_spm, data=hyl_vals)
        mdl = mdl if ltnt_vals is None else condition(mdl, data=ltnt_vals)
        mdl = mdl if meas_vals is None else condition(mdl, data=meas_vals)

        def sampler(rng):
            seeded = seed(mdl, rng)
            tr = trace(seeded).get_trace(test.config, test.conditions, pri, self.param_vals)
            samples = {site: tr[site]['value'] for site in tr}
            return dict((k, samples[k]) for k in keep_sites) if keep_sites is not None else samples

        return jax.vmap(sampler, axis_size=num_samples)(rand.split(rng_key, num_samples))

    def sample_new(self, rng_key: rand.key, test: ReliabilityTest, num_samples: tuple = (1,), keep_sites: list = None,
                   conditionals: dict = None):
        # Now handle any conditioning where sample sites are fixed to values
        #mdl = self.test_spm if conditionals is None else condition(self.test_spm, data=conditionals)
        # Convert any keep_sites that need more complex node names
        sites = keep_sites.copy()
        for i, site in enumerate(sites):
            if site in self.test_measurements:
                sites[i] = f'{next(iter(test.config))}_{site}'

        def sampler(rng, set_vals):
            mdl = self.test_spm if set_vals is None else condition(self.test_spm, data=set_vals)
            seeded = seed(mdl, rng)
            tr = trace(seeded).get_trace(test.config, test.conditions, self.hyl_beliefs, self.param_vals)
            samples = {site: tr[site]['value'] for site in tr}
            return dict((k, samples[k]) for k in sites) if sites is not None else samples

        wrapped = sampler
        size = 1
        for i, dim in enumerate(num_samples):
            size *= dim
            wrapped = jax.vmap(wrapped, 0, 0)
        keys = jnp.reshape(rand.split(rng_key, size), num_samples)

        return wrapped(keys, conditionals)

    def hyl_logp(self, rng_key: rand.key, test: ReliabilityTest, hyl_vals: dict):
        mdl = condition(self.test_spm, data=hyl_vals)

        seeded = seed(mdl, rng_key)
        tr = trace(seeded).get_trace(test.config, test.conditions, self.hyl_beliefs, self.param_vals)
        logp = 0
        for site in hyl_vals:
            logp += tr[site]['fn'].log_prob(hyl_vals[site])
        return logp

    def logp(self, rng_key: rand.key, test: ReliabilityTest, site_vals: dict, conditional: dict = None):

        def logp(rng, vals, cond):
            mdl = self.test_spm if cond is None else condition(self.test_spm, data=cond)
            seeded = seed(mdl, rng)
            tr = trace(seeded).get_trace(test.config, test.conditions, self.hyl_beliefs, self.param_vals)
            lp = 0
            for site in vals:
                lp += tr[site]['fn'].log_prob(vals[site])
            return lp

        shape = next(iter(site_vals.values())).shape
        num_vals = 1 if len(shape) == 0 else shape[0]
        #cond_vals = [None for _ in range(num_vals)] if conditional is None else conditional
        cond_vals = conditional
        return jax.vmap(logp, axis_size=num_vals)(rand.split(rng_key, num_vals), site_vals, cond_vals)

    def logp_new(self, rng_key: rand.key, test: ReliabilityTest, site_vals: dict, conditional: dict, dims: tuple):

        def logp(rng, vals, cond):
            mdl = self.test_spm if cond is None else condition(self.test_spm, data=cond)
            seeded = seed(mdl, rng)
            tr = trace(seeded).get_trace(test.config, test.conditions, self.hyl_beliefs, self.param_vals)
            lp = 0
            for site in vals:
                lp += jnp.sum(tr[site]['fn'].log_prob(vals[site]))
            return lp

        wrapped = logp
        size = 1
        for dim in dims:
            size *= dim
            wrapped = jax.vmap(wrapped, 0, 0)

        keys = jnp.reshape(rand.split(rng_key, size), dims)
        return wrapped(keys, site_vals, conditional)