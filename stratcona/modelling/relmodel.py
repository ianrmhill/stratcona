# Copyright (c) 2025 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import jax.random as rand

from numpyro.handlers import seed, trace, condition

from typing import Callable
from dataclasses import dataclass
from frozendict import frozendict
from functools import partial


class ReliabilityRequirement:
    metric: Callable
    quantile: float
    target_lifespan: float

    def __init__(self, metric, quantile, target_lifespan):
        self.type = metric
        self.quantile = quantile
        self.target_lifespan = target_lifespan


@dataclass(frozen=True)
class ExpDims:
    name: str
    lot: int
    chp: int
    dev: frozendict[str, int]

    def __init__(self, name, lot, chp, **dev):
        object.__setattr__(self, 'name', name)
        object.__setattr__(self, 'lot', lot)
        object.__setattr__(self, 'chp', chp)
        object.__setattr__(self, 'dev', frozendict({y: dev[y] for y in dev}))

    def __members(self):
        return self.name, self.lot, self.chp, self.dev

    def __hash__(self):
        return hash(self.__members())

    def __eq__(self, other):
        return type(self) is type(other) and self.__members() == other.__members()


class TestDef:
    name: str
    # Note that the tuple implementation means order matters when comparing tests for equality, as the test iteration
    # order impacts execution paths and so must match between JIT-compiled executions
    dims: tuple[ExpDims, ...]
    conds: dict[str, dict[str, float | int]]

    def __init__(self, name: str, dims: dict[str, dict[str, float | int]] | tuple[ExpDims],
                 conditions: dict[str, dict[str, float | int]]):
        self.name = name
        if type(dims) is dict:
            self.dims = tuple(ExpDims(e, **dims[e]) for e in dims)
        else:
            self.dims = dims
        self.conds = conditions

    def __members(self):
        return self.name, self.dims, self.conds

    def __eq__(self, other):
        return type(self) is type(other) and self.__members() == other.__members()


class ReliabilityModel:
    name: str
    spm: Callable
    observes: tuple[str]
    predictors: tuple[str]
    obs_per_chp: dict[str, int]
    fail_criteria: tuple[str]
    hyls: tuple[str]
    hyl_beliefs: dict[dict[str: float]]
    hyl_info: dict[dict]

    def __init__(self, name, spm, param_vals, hyl_sites, prior, hyl_info, ltnt_sites,
                 ltnt_subsample_sites, obs_sites, meas_counts, obs_noise, pred_sites, fail_sites):
        self.name = name
        self.spm = spm

        self.param_vals = param_vals

        self.hyls = hyl_sites
        self.hyl_beliefs = prior
        self.hyl_info = hyl_info

        self.ltnts = ltnt_sites
        self.ltnt_subsamples = ltnt_subsample_sites

        self.observes = obs_sites
        self.obs_per_chp = meas_counts
        self.obs_noise = obs_noise

        self.predictors = pred_sites
        self.fail_criteria = fail_sites

        self.i_s_override = None
        self.y_s_override = None

    def __setattr__(self, name, value):
        # Because the SPM function depends on the model itself, the model structure, parameter values, and priors
        # all get baked into the JIT-compiled function. If the priors or parameter values are changed, the compilation
        # caches need to be cleared so that the old values, baked into the compiled functions, are not still used
        if name == 'hyl_beliefs' or name == 'param_vals':
            jax.clear_caches()
        super().__setattr__(name, value)

    @partial(jax.jit, static_argnames=['self', 'test_dims', 'batch_dims', 'keep_sites', 'full_trace', 'compute_predictors'])
    def sample(self, rng_key: rand.key, test_dims: tuple[ExpDims, ...], test_conds: dict, batch_dims: tuple = (),
               keep_sites: tuple = None, conditionals: dict = None, full_trace=False, compute_predictors=False):
        # Convert any keep_sites that need more complex node names
        if keep_sites is not None:
            sites = []
            for i, site in enumerate(keep_sites):
                if site in self.observes or site in self.predictors:
                    for tst in test_dims:
                        sites.append(f'{tst.name}_{site}')
                elif site in self.ltnts or site in self.ltnt_subsamples:
                    for tst in test_dims:
                        if '_lot' in site or '_chp' in site:
                            sites.append(f'{tst.name}_{site}')
                        else:
                            for obs in self.observes:
                                sites.append(f'{tst.name}_{obs}_{site}')
                elif site in self.ltnt_subsamples:
                    pass
                else:
                    sites.append(site)
        else:
            sites = None

        priors = self.hyl_beliefs

        def sampler(rng, set_vals):
            mdl = self.spm if set_vals is None else condition(self.spm, data=set_vals)
            seeded = seed(mdl, rng)
            tr = trace(seeded).get_trace(test_dims, test_conds, priors, self.param_vals, compute_predictors=compute_predictors)
            samples = {site: tr[site] if full_trace else tr[site]['value'] for site in tr}
            return dict((k, samples[k]) for k in sites) if sites is not None else samples

        wrapped = sampler
        size = 1
        for i, dim in enumerate(batch_dims):
            size *= dim
            wrapped = jax.vmap(wrapped, 0, 0)
        keys = jnp.reshape(rand.split(rng_key, size), batch_dims)

        return wrapped(keys, conditionals)

    # Note that jax jit implicitly handles the dict arguments. Each unique set of keys for a dict arg leads to a
    # recompilation, so in reality almost all args are at least partially static
    @partial(jax.jit, static_argnames=['self', 'test_dims', 'batch_dims', 'sum_lps'])
    def logprob(self, rng_key: rand.key, test_dims: tuple[ExpDims, ...], test_conds: dict, site_vals: dict,
                conditional: dict | None, batch_dims: tuple = (), sum_lps=True):

        def get_log_prob(rng, vals, cond):
            mdl = self.spm if cond is None else condition(self.spm, data=cond)
            seeded = seed(mdl, rng)
            tr = trace(seeded).get_trace(test_dims, test_conds, self.hyl_beliefs, self.param_vals)
            lp = 0 if sum_lps else {}
            for site in vals:
                if sum_lps:
                    lp += jnp.sum(tr[site]['fn'].log_prob(vals[site]))
                else:
                    lp[site] = tr[site]['fn'].log_prob(vals[site])
            return lp

        wrapped = get_log_prob
        size = 1
        for dim in batch_dims:
            size *= dim
            wrapped = jax.vmap(wrapped, 0, 0)

        keys = jnp.reshape(rand.split(rng_key, size), batch_dims)
        return wrapped(keys, site_vals, conditional)
