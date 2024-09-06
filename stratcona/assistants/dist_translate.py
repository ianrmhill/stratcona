# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import jax.numpy as jnp
import numpyro.distributions as npyro_dists
import scipy.stats.distributions as scipy_dists

def npyro_to_scipy(dist_type):
    fit_kwargs = {}
    unity = lambda x: x
    match dist_type:
        case npyro_dists.Normal:
            dist = scipy_dists.norm
            prm_order = ['loc', 'scale']
            prm_transforms = [unity, unity]
        case npyro_dists.LogNormal:
            dist = scipy_dists.lognorm
            prm_order = ['scale', None, 'loc']
            loc_t = lambda loc: jnp.log(loc)
            prm_transforms = [unity, None, loc_t]
            fit_kwargs['floc'] = 0
        case npyro_dists.HalfNormal:
            dist = scipy_dists.halfnorm
            prm_order = [None, 'scale']
            prm_transforms = [None, unity]
            # Fix the location parameter to zero
            fit_kwargs['floc'] = 0
        case npyro_dists.TruncatedNormal:
            dist = scipy_dists.truncnorm
            prm_order = [None, None, 'loc', 'scale']
            prm_transforms = [None, None, unity, unity]
        case _:
            raise Exception(f'Unmapped distribution type {dist_type}!')
    return dist, prm_order, prm_transforms, fit_kwargs


def pymc_to_scipy(dist_type):
    match dist_type:
        case 'normal':
            return 'norm'
        case 'truncated_normal':
            return 'truncnorm'
        case _:
            return dist_type


def pymc_to_numpy(dist_type):
    match dist_type:
        case _:
            return dist_type


def convert_to_categorical(dist, prms, num_samples: int = 30_000):
    # Count the number of occurrences of each value then normalize to get the posterior probabilities
    np_dist = pymc_to_numpy(dist)
    sampled = getattr(np.random, np_dist)(**prms, size=num_samples)
    return np.bincount(sampled) / num_samples
