# Copyright (c) 2025 Ian Hill
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import time as t
import datetime
import json
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import jax.random as rand

from multiprocessing import Pool
from progress.bar import Bar
from functools import partial
from typing import Callable
from inspect import signature
from inspect import Parameter

from matplotlib import pyplot as plt

from stratcona.engine.inference import int_out_v


NATS_TO_BITS = jnp.log2(jnp.e)
# Machine precision is defaulted to 32 bits since most instruments don't have 64 bit precision and/or noise floors. Less
# precise but smaller entropy values makes things nicer to deal with.
CARDINALITY_MANTISSA_32BIT = float(2 ** 23)
CARDINALITY_MANTISSA_64BIT = float(2 ** 53)
# Marginal probability below which a sample observed 'y' will be excluded from EIG computation
LOW_PROB_CUTOFF = 1e-20
# Gap between trace samples to reduce memory and performance impact
TR_GAP = 20


def lp_mx(x: jnp.ndarray, a: float, b: float):
    """The log-probability distribution function of a continuous uniform distribution on the interval [a, b]."""
    lp = jnp.log(1 / (b - a))
    return jnp.where(jnp.less_equal(a, x) & jnp.less_equal(x, b), lp, -jnp.inf)


def entropy(samples, lp_func, p_args=None, in_bits=False,
            limiting_density_range: tuple = None, precision=CARDINALITY_MANTISSA_32BIT, d=1):
    """
    This procedure assumes the samples are randomly drawn from the prior, thus distributed according to the prior.

    Parameters
    ----------
    samples
    lp_func
    p_args
    in_bits
    limiting_density_range
    precision

    Returns
    -------

    """

    # Entropy is predicated on iterating over all possible values that theta can take, thus the overall probability
    # summed across all samples of theta is 1. In the continuous case or to compute via sampling, we need to normalize
    # with respect to the probability sum across all the samples of theta.
    lp_f = lp_func(samples, *p_args) if p_args else lp_func(samples)
    # Samples can either be an array or dictionary of equal sized arrays
    n = samples.size if type(samples) != dict else samples[next(iter(samples))].size

    if limiting_density_range is not None:
        a, b = limiting_density_range[0], limiting_density_range[1]
        if type(samples) != dict:
            lp_u = lp_mx(samples, a, b)
        else:
            lp_u = jnp.zeros_like(samples[next(iter(samples))])
            for site in samples:
                lp_u += lp_mx(samples[site], a, b)
        # The integral to compute the LDDP is bounded by the m(x) uniform distribution, thus all samples outside the
        # range of integration are discarded for the purposes of computing the LDDP
        oob_samples = jnp.isinf(lp_u)
        oob_count = jnp.count_nonzero(oob_samples)
        lp_f = jnp.where(oob_samples, 0, lp_f - lp_u)
        n -= oob_count

    h = -jnp.sum(lp_f) / n

    # Try to inform the user on how well the samples captured the probability distribution
    # Metric for detecting whether the full high probability region of the distribution was sampled well
    if limiting_density_range is not None:
        # Inform the user if the bounded region u may be too small
        oob_p = oob_count / n
        if oob_p > 0.01:
            print(f"{round(oob_p * 100, 2)}% of the sampled PDF outside LDDP bounds. You may want to extend bounds.")

        # For limiting density, the entropy is log(N) + h, where N is the number of discrete points in the interval
        # [a, b]. For our floating point representations, we assume that the exponent bits determine a and b, then
        # N is the number of possible values for the mantissa bits, i.e., the cardinality. This way N doesn't change
        # regardless of if our random variable support is [0, 1] or [-1e10, 1e10] and represents a physical property
        # of our measurements in terms of their precision once converted to floating point representations.
        # Note that 'h' will always be negative until the precision has been added in the limiting density formulation.
        h += d * jnp.log(precision)

    # Support representing entropy in units of 'nats' or 'bits'
    return h if not in_bits else h * NATS_TO_BITS


def w_quantile(s, w, q, y_dim=2):
    """
    This procedure is based on the 'wquantiles' library but adapted for JAX numpy and vectorized.

    Parameters
    ----------
    s - sampled values
    w - weights for the sampled values, shape should match 's'
    q - the requested quantile to compute
    y_dim - vectorization dimension, computing a different QX%-LBCI for each y value along that dimension

    Returns
    -------
    The quantile computed from the weighted samples
    """
    s1 = jnp.moveaxis(s, y_dim, -1)
    w1 = jnp.moveaxis(w, y_dim, -1)
    s2 = jnp.reshape(s1, (-1, s.shape[y_dim]))
    w2 = jnp.reshape(w1, (-1, w.shape[y_dim]))

    # Samples are shared across values of y, so only need to sort for one y, all others will match
    order = jnp.argsort(s2[:, 0], axis=0)
    s_srt = s2[order]
    w_srt = w2[order]

    w_cumulative = jnp.cumsum(w_srt, axis=0)
    # Subtract half the probability of the value to get the CDF midpoint for each sample, normalize CDF to [0, 1]
    # Normalization term, the sum of all weights, was already computed by cumsum as last element of array
    bins = (w_cumulative - (0.5 * w_srt)) / w_cumulative[-1]
    # Don't map over x=q, but axis 1 of bins and s_srt correspond to different y observations. Out axis is 0 as interp
    # with a single float x=q will only produce a 1D output array, collapsing axis 0 in bins and s_srt
    q_func_vect = jax.vmap(jnp.interp, (None, 1, 1), 0)
    return q_func_vect(q, bins, s_srt)


@jax.jit
def h_x_g_y(lw, lp_x):
    """
    Using importance sampling from the prior, H[X|y] = (1/sum[p(x|y)/p(x)]) * sum[(p(x|y)/p(x))*(-lp_x_g_y)].
    Substituting p(x|y) = p(y|x)p(x) / p(y) and using w = p(x|y) / p(y)
    H[X|y] = (1/sum[w]) * sum[w * -log(p(x) * w)]
    """
    lp_x_ty = jnp.expand_dims(lp_x, axis=1)
    w = jnp.exp(lw)
    return (1 / jnp.sum(w, axis=0)) * jnp.sum(w * -(lp_x_ty + lw), axis=0)


@jax.jit
def eig(ig):
    return jnp.sum(ig) / ig.size


@jax.jit
def qx_lbci(w, z, q):
    # Z expected to have dimensions (n_x, n_z)
    z_ty = jnp.repeat(jnp.expand_dims(z, axis=2), w.shape[1], axis=2)
    # w expected to have dimensions (n_x, n_y)
    w_t = jnp.repeat(jnp.expand_dims(w, axis=1), z.shape[1], axis=1)
    return w_quantile(z_ty, w_t, q)


@partial(jax.jit, static_argnames=['num_bins'])
def qx_hdcr(w, z, q: int | float, num_bins: int = 1000):
    """
    The highest density interval can be useful for situations in which we aren't worried about liability of edge cases
    with poor reliability and prefer to find a general estimate of the likely lifespan for a product.
    """
    # Z expected to have dimensions (n_x, n_z)
    z_ty = jnp.repeat(jnp.expand_dims(z, axis=2), w.shape[1], axis=2)
    # w expected to have dimensions (n_x, n_y)
    w_t = jnp.repeat(jnp.expand_dims(w, axis=1), z.shape[1], axis=1)
    # We bin the samples into intervals, then keep adding the intervals with the highest counts until we pass the target
    bins = jnp.linspace(jnp.min(z_ty, axis=0), jnp.max(z_ty, axis=0), num_bins)
    bin_len = bins[1] - bins[0]
    densities, intervals = jnp.histogram(z_ty, bins, density=True)
    # Sort the bins from highest density to lowest (and their corresponding intervals)
    i_order = jnp.flip(jnp.argsort(densities))
    sorted_density = densities[i_order]
    # Add the largest bins until the interval size is surpassed
    i = 0
    summed = jnp.cumsum(sorted_density, axis=0)
    while jnp.sum(sorted_density[:i] * bin_len) < (q / 100):
        i += 1
    # Now determine the size of the HDCR
    region_size = bin_len * i
    # In this compiled version no sanity checks are performed for speed and compilation reasons
    return region_size


@partial(jax.jit, static_argnames=['spm', 'd_dims', 'batch_dims', 'utility', 'fd_dims'])
def eval_u_of_d(k, spm, d_dims, d_conds, x_s, batch_dims, utility, lp_x, h_x, fd_dims, fd_conds):
    n_y, n_v, n_x = batch_dims
    k, kv, ky, ku, kz, kd = rand.split(k, 6)
    # Sample observations from joint prior y~p(x,v,y|d), keeping y independent of the already sampled x and v values
    y_s = spm.sample_new(ky, d_dims, d_conds, batch_dims=(n_y,), keep_sites=spm.observes)

    # Compute the log likelihoods of each y given x via our specialized resampling algorithm to marginalize across
    # the aleatoric uncertainty v
    batch_dims = (n_x, n_v, n_y)
    y_noise = spm.obs_noise
    lp_y_g_x, v_marg_alg_stats = int_out_v(kv, spm, batch_dims, d_dims, d_conds, x_s, y_s, y_noise)

    # Marginalize across epistemic uncertainty axis 'x'
    lp_y = logsumexp(lp_y_g_x, axis=0, keepdims=True) - jnp.log(n_x)
    # Now can compute the importance weights
    lw_z = lp_y_g_x - lp_y
    # Properties of algebra during the lp_y_g_x - lp_y step mean w_z_norm will always be the value of n_x
    w_z = jnp.exp(lw_z)

    # Now compute summary statistics for each sample 'y'
    metrics = {}
    if hasattr(utility, 'keywords'):
        ms = [prm for prm in signature(utility).parameters if prm not in utility.keywords]
    else:
        ms = [prm for prm in signature(utility).parameters]
    for m in ms:
        # Start by supporting IG and QX%-LBCI metrics, ideally QX%-HDCR too
        match m:
            case 'ig':
                h_xgy = h_x_g_y(lw_z, lp_x)
                metrics[m] = h_x - h_xgy
            case 'qx_lbci':
                n_z = n_v
                x_s_tz = {x: jnp.repeat(jnp.expand_dims(x_s[x], axis=1), n_z, axis=1) for x in x_s}
                z_s = spm.sample_new(kz, fd_dims, fd_conds, (n_x, n_z), keep_sites=spm.predictors,
                                     conditionals=x_s_tz, compute_predictors=True)
                metrics[m] = qx_lbci(w_z, z_s['field_lifespan'], 0.01)
            case 'qx_hdcr':
                n_z = n_v
                x_s_tz = {x: jnp.repeat(jnp.expand_dims(x_s[x], axis=1), n_z, axis=1) for x in x_s}
                z_s = spm.sample_new(kz, fd_dims, fd_conds, (n_x, n_z), keep_sites=spm.predictors,
                                     conditionals=x_s_tz, compute_predictors=True)
                metrics[m] = qx_hdcr(w_z, z_s['field_lifespan'], 0.1)
            case 'p_y':
                # NOTE: Should almost never need p_y directly since the samples y_s are already distributed
                #       according to p(y)
                # Always have p_y sum to 1 to make calculation of summary statistics easier
                lp_y_normed = lp_y - logsumexp(lp_y)
                metrics[m] = jnp.exp(lp_y_normed).flatten()
            case _:
                raise Exception('Unsupported metric requested')

    # Now compute the utility summary statistic for 'd' based on the metrics for each 'y' sample
    return utility(**metrics)


def pred_bed_apr25(rng_key, d_sampler, n_d, n_y, n_v, n_x, spm, utility=eig, field_d=None):
    k, kd, kx = rand.split(rng_key, 3)
    perf_stats = {}
    # Get the first proposal experiment design, need to sample here to get dummy input to x_s sample and lp_x logp
    d = d_sampler(kd)
    # We sample all x~p(x) here since hyper-parameters are independent of the test d, the observations y, and the predictors z
    x_s = spm.sample_new(kx, d.dims, d.conds, batch_dims=(n_x,), keep_sites=spm.hyls)
    # Compute the prior entropy once, but only if it's needed for EIG estimation
    if 'ig' in signature(utility).parameters:
        lp_x = spm.logp_new(kx, d.dims, d.conds, site_vals=x_s, conditional=None, batch_dims=(n_x,))
        # Using importance sampling from prior: H[X] = (1/n_x) * SUM[-lp_x]
        h_x = jnp.sum(-lp_x) / n_x
    else:
        lp_x, h_x = 0, 0

    # We loop over sample test designs instead of vectorizing since test designs can modify dimensionality
    bar = Bar('Evaluating possible designs', max=n_d)
    us = []
    for _ in range(n_d):
        u = eval_u_of_d(k, spm, d.dims, d.conds, x_s, (n_y, n_v, n_x), utility, lp_x, h_x, field_d.dims, field_d.conds)
        us.append({'design': d, 'utility': u.block_until_ready()})
        bar.next()
        # Now update the test design for the next iteration, the final design (index n_d) is not used
        d = d_sampler(kd)
    bar.finish()
    return us, perf_stats
