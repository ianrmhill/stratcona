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
    spm: Callable
    observes: list[str]
    predictors: list[str]
    obs_per_chp: dict[int]
    fail_criteria: list[str]
    hyls: dict[str]
    hyl_beliefs: dict[dict[str: float]]
    hyl_info: dict[dict]

    def __init__(self, name, spm, param_vals, hyl_sites, prior, hyl_info, ltnt_sites,
                 ltnt_subsample_sites, obs_sites, meas_counts, pred_sites, fail_sites):
        self.name = name
        self.spm = spm

        self.param_vals = param_vals

        self.hyls = hyl_sites
        self.hyl_beliefs = prior
        self.hyl_info = hyl_info

        self.ltnts = ltnt_sites
        self._ltnt_subsamples = ltnt_subsample_sites

        self.observes = obs_sites
        self.obs_per_chp = meas_counts

        self.predictors = pred_sites
        self.fail_criteria = fail_sites

        self.i_s_override = False
        self.y_s_override = False
        self.i_s_custom = None
        self.y_s_custom = None

    def validate_model(self):
        # TODO
        pass

    def sample(self, rng_key: rand.key, test: ReliabilityTest, num_samples: tuple = (), keep_sites: list = None,
                   conditionals: dict = None, full_trace=False, alt_priors=None):
        # Convert any keep_sites that need more complex node names
        if keep_sites is not None:
            sites = []
            for i, site in enumerate(keep_sites):
                if site in self.observes or site in self.predictors:
                    for tst in test.config:
                        sites.append(f'{tst}_{site}')
                elif site in self.ltnts or site in self._ltnt_subsamples:
                    for tst in test.config:
                        if '_dev' in site:
                            for obs in self.observes:
                                sites.append(f'{tst}_{obs}_{site}')
                        else:
                            sites.append(f'{tst}_{site}')
                else:
                    sites.append(site)
        else:
            sites = None

        priors = self.hyl_beliefs if alt_priors is None else alt_priors

        def sampler(rng, set_vals):
            mdl = self.spm if set_vals is None else condition(self.spm, data=set_vals)
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
            mdl = self.spm if cond is None else condition(self.spm, data=cond)
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
