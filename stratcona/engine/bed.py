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
CARDINALITY_MANTISSA_32BIT = float(2 ** 24)
CARDINALITY_MANTISSA_64BIT = float(2 ** 54)
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


@partial(jax.jit, static_argnames=['y_dim'])
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
    return q_func_vect(1 - q, bins, s_srt)


@jax.jit
def h_x_g_y(lw, lp_x):
    """
    Using importance sampling from the prior, H[X|y] = (1/sum[p(x|y)/p(x)]) * sum[(p(x|y)/p(x))*(-lp_x_g_y)].
    Substituting p(x|y) = p(y|x)p(x) / p(y) and using w = p(y|x) / p(y)
    H[X|y] = (1/sum[w]) * sum[w * -log(p(x) * w)]
    """
    lp_x_ty = jnp.expand_dims(lp_x, axis=1)
    w = jnp.exp(lw)
    return (1 / jnp.sum(w, axis=0)) * jnp.sum(w * -(lp_x_ty + lw), axis=0)


@jax.jit
def eig(ig):
    return jnp.sum(ig) / ig.size


@jax.jit
def qx_lbci(z, w, q):
    # Z expected to have dimensions (n_x, n_z)
    z_ty = jnp.repeat(jnp.expand_dims(z, axis=2), w.shape[1], axis=2)
    # w expected to have dimensions (n_x, n_y)
    w_t = jnp.repeat(jnp.expand_dims(w, axis=1), z.shape[1], axis=1)
    return w_quantile(z_ty, w_t, q)


@partial(jax.jit, static_argnames=['n_bins'])
def qx_hdcr_width(z, w, q, n_bins):
    """
    Compilable function to compute the region width of the QX%-HDCR (highest density credible region for some quantile)
    given sample values z and sample weights w. Pay careful attention to the dimensions of z and w as they are
    unintuitive.

    Parameters
    ----------
    z: (n_x, n_z) - A set of n_x distributions represented as n_z samples
    w: (n_x, n_y) - A set of weightings for each of the n_x distributions per batch dimension n_y
    q: float - The quantile cutoff for the HDCR (i.e., the X in Q<X>%-HDCR)
    n_bins: int - The number of buckets to group the z samples into to model the overall weighted distributions

    Returns
    -------
    widths - (n_y,) - The HDCR total region width for each batch element
    """
    # Duplicate the weights to apply to all n_z samples that compose each of the n_x distributions
    w_t = jnp.repeat(jnp.expand_dims(w, axis=1), z.shape[1], axis=1)
    # Generate the weighted histogram of values
    densities, bin_edges = jax.vmap(jnp.histogram, (None, None, None, 2), 1)(z, n_bins, None, w_t)
    densities = densities / jnp.sum(densities, axis=0, keepdims=True)
    # Sort the PDF histogram bins from highest density to lowest
    sorted_density = jnp.flip(jnp.sort(densities, axis=0), axis=0)
    # Find the minimum number of bins required to capture q% of the total distribution
    cum = jnp.cumsum(sorted_density, axis=0)
    ind = jax.vmap(jnp.searchsorted, (1, None), 0)(cum, q)
    # Now determine the size of the HDCR, no analysis of appropriate number of bins for speed and compilation reasons
    return (bin_edges[1, :] - bin_edges[0, :]) * ind


@partial(jax.jit, static_argnames=['spm', 'd_dims', 'batch_dims', 'utility', 'fd_dims', 'predictor'])
def eval_u_of_d(k, spm, d_dims, d_conds, x_s, batch_dims, utility, lp_x, h_x, fd_dims, fd_conds, predictor):
    n_y, n_v, n_x = batch_dims
    k, kv, ky, ku, kz, kd = rand.split(k, 6)
    # Sample observations from joint prior y~p(x,v,y|d), keeping y independent of the already sampled x and v values
    y_s = spm.sample_new(ky, d_dims, d_conds, batch_dims=(n_y,), keep_sites=spm.observes)

    # Compute the log likelihoods of each y given x via our specialized resampling algorithm to marginalize across
    # the aleatoric uncertainty v
    batch_dims = (n_x, n_v, n_y)
    y_noise = spm.obs_noise
    # Have to map computation to avoid gigantic memory allocations
    if n_v > 1:
        marg_v_part = partial(int_out_v, spm=spm, batch_dims=(n_x, n_v, 1), test_dims=d_dims, test_conds=d_conds, x_s=x_s, y_noise=y_noise)

        def v_part(args):
            return marg_v_part(rng_key=args['k'], y_s=args['y'])[0]

        keys = rand.split(kv, next(iter(y_s.values())).shape[0])
        lp_y_g_x = jax.lax.map(v_part, {'k': keys, 'y': y_s}, batch_size=2)
    else:
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
                # FIXME: Don't implicitly use n_v for n_z, the user may want control over n_z and n_v may not be
                #        appropriate in all circumstances
                n_z = n_v
                x_s_tz = {x: jnp.repeat(jnp.expand_dims(x_s[x], axis=1), n_z, axis=1) for x in x_s}
                z_s = spm.sample_new(kz, fd_dims, fd_conds, (n_x, n_z), keep_sites=spm.predictors,
                                     conditionals=x_s_tz, compute_predictors=True)
                metrics[m] = qx_lbci(z_s[f'field_{predictor}'], w_z, 0.99)
            case 'qx_hdcr_width':
                n_z = n_v
                x_s_tz = {x: jnp.repeat(jnp.expand_dims(x_s[x], axis=1), n_z, axis=1) for x in x_s}
                z_s = spm.sample_new(kz, fd_dims, fd_conds, (n_x, n_z), keep_sites=spm.predictors,
                                     conditionals=x_s_tz, compute_predictors=True)
                metrics[m] = qx_hdcr_width(z_s[f'field_{predictor}'], w_z, 0.9, n_bins=100)
            case 'test_duration':
                ew = jnp.zeros((len(d_dims), n_y))
                for i, e in enumerate(d_dims):
                    ew = ew.at[i].set(jnp.max(y_s[f'{e.name}_lttf'], axis=(1, 2, 3)))
                metrics[m] = jnp.max(ew, axis=0)
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


def pred_bed_apr25(rng_key, d_sampler, n_d, n_y, n_v, n_x, spm, utility=eig, field_d=None, predictor='lifespan'):
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
        k, kud = rand.split(k)
        u = eval_u_of_d(kud, spm, d.dims, d.conds, x_s, (n_y, n_v, n_x), utility, lp_x, h_x, field_d.dims, field_d.conds, predictor)
        us.append({'design': d, 'utility': u})
        #print(f"\nEwidth: {u['e_qx_hdcr_width'].block_until_ready()}\n")
        bar.next()
        # Now update the test design for the next iteration, the final design (index n_d) is not used
        d = d_sampler(kd)
    bar.finish()
    return us, perf_stats
