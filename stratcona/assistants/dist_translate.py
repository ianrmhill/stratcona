# Copyright (c) 2025 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import numpyro.distributions as npyro_dists
import jax.numpy as jnp
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


def convert_to_categorical(dist, prms, num_samples: int = 30_000):
    """
    Fits a categorical distribution to some discrete
    """
    # Count the number of occurrences of each value then normalize to get the posterior probabilities
    sdist = npyro_to_scipy(dist)[0]
    sampled = sdist.sample(**prms, size=num_samples)
    return jnp.bincount(sampled) / num_samples
