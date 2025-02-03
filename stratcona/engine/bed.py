# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import time as t
import datetime
import json
import warnings
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import jax.random as rand
import wquantiles
import pandas as pd

from multiprocessing import Pool
from functools import partial
from typing import Callable
from inspect import signature
from inspect import Parameter

from matplotlib import pyplot as plt

from stratcona.modelling.relmodel import ReliabilityTest


__all__ = ['eig_smc_refined', 'bed_runner']

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


def information_gain(theta_sampler, p_pri, p_post, m=1e5, in_bits=False, limiting_density=False):
    """
    Placeholder for now, may be nice to have in the future for comparing the actual information gain to the expected
    gain after running experiments.
    """
    h_theta = entropy(theta_sampler, p_pri, m, None, in_bits, limiting_density)
    h_theta_given_y = entropy(theta_sampler, p_post, m, None, in_bits, limiting_density)
    print(f"H: {h_theta}, H given y: {h_theta_given_y}")
    return h_theta - h_theta_given_y


def one_obs_process(n_i, obs_n, y_i, m, i_s: Callable, y_s: Callable, lp_i: Callable, lp_y: Callable, lp_i_avg, lp_y_avg, traces, f_l, f_l_dims , metric_region_size):
    """

    Parameters
    ----------
    n_i: Index of the current observation space sample
    obs_n: Array containing all observation space samples
    y_i: Index of the current partial observation sample
    m: Number of latent space samples to use
    i_s: Latent space sampler function
    y_s: Partial observation sampler function
    lp_i
    lp_y
    lp_i_avg
    lp_y_avg
    traces: Bool indicating whether to collect a trace for the computation
    f_l: Predictive lifespan function
    f_l_dims
    metric_region_size

    Returns
    -------

    """
    ### Setup ###
    y = obs_n
    lp_lkly_store = jnp.zeros((m,))
    lp_pri_store = jnp.zeros((m,))
    lifespan_store = jnp.zeros((m, *f_l_dims))
    lp_cond_store = jnp.zeros((y_i + 1,))
    ig_final, marg, metric = 0.0, 0.0, 0.0
    tr_flag = traces and (n_i + 1) % TR_GAP == 0
    ig_traces = jnp.zeros((int(m / TR_GAP),)) if tr_flag else None

    ### Computation ###
    y[y_i] = y_s()

    for m_i in range(m):
        i = i_s()
        lp_pri = lp_i(*i) - lp_i_avg
        lp_pri_store[m_i] = lp_pri
        for y_j in range(y_i + 1):
            lp_cond_store[y_j] = lp_y(*i, *y[y_j]) - lp_y_avg
        # Persist the log importance weighted likelihood
        lp_lkly_store[m_i] = jnp.sum(lp_cond_store[:y_i + 1]) + lp_pri
        # Compute the product lifespan for the given latent variable space sample
        if f_l is not None:
            lf = f_l(*i)
            #lifespan_store[m_i] = f_l(*i)[0]
            lifespan_store[m_i] = lf

        # Compute the IG on the final iteration and for any optionally traced samples
        end = m_i + 1
        if end == m or (traces and (n_i + 1) % TR_GAP == 0 and end % TR_GAP == 0):
            p_pri = jnp.exp(lp_pri_store[:end])
            p_lkly = jnp.exp(lp_lkly_store[:end])
            norm = jnp.sum(p_pri)
            marg = jnp.sum(p_lkly) / norm
            # We exclude extremely low probability observed 'y' samples since they shouldn't affect the EIG
            # in theory, however the computation when calculating p_post and lp_post can become unstable
            if marg < LOW_PROB_CUTOFF:
                marg = 0.0
                # Skip so that we don't take log of 0 or divide by 0, IG for the sample will remain 0
                continue
            # Compute the un-normalized prior entropy (but with flipped sign)
            h_pri = jnp.sum(p_pri * lp_pri_store[:end])
            # Compute the un-normalized posterior entropy (but with flipped sign)
            lp_post = lp_lkly_store[:end] - jnp.log(marg)
            p_post = p_lkly / marg
            h_post = jnp.sum(p_post * lp_post)
            # Finally, compute the normalized information gain for the sampled 'y'
            ig = (h_post - h_pri) / norm
            if tr_flag:
                ig_traces[int(m_i / TR_GAP)] = ig
            if end == m:
                # Separate variable required in case we hit 'continue' on the final loop iteration
                ig_final = ig
                # X% quantile estimation
                if f_l is not None:
                    # NOTE: Numpy 2.0's weighted quantile estimation only works with the inverted CDF approach, I may
                    #       need to write a custom estimator if I want to use the continuous approaches or support 0
                    #       prob samples: see https://arxiv.org/abs/2304.07265
                    #metric = np.quantile(lifespan_store, 1 - (metric_region_size / 100), weights=np.exp(lp_lkly_store))
                    # TODO: Validate this wquantiles library or write my own weighted quantile estimation algorithm(s)
                    lkly_weights = jnp.exp(lp_lkly_store)
                    lifespans = lifespan_store.copy()
                    if len(f_l_dims) > 0:
                        lkly_weights = jnp.repeat(lkly_weights, int(lifespan_store.size / lkly_weights.size))
                        lifespans = lifespans.flatten()
                    metric = wquantiles.quantile(lifespans, lkly_weights, 1 - (metric_region_size / 100))

    return y, ig_final, marg, metric, ig_traces


def eig_smc_refined(n, m, i_s, y_s, lp_i, lp_y,
                    ys_per_obs: int = 1, p_y_stabilization=True, p_i_stabilization=False,
                    compute_traces: bool = False, resample_rng=None, multicore=True, rig=None,
                    f_l=None, metric_trgt=None, credible_region_size=99):
    """
    Core EIG (expected information gain) computation function that produces an estimate using sampling. This sampler
    uses importance sampling and has the benefit of being exact for finite discrete models if the samplers i_s and y_s
    are constructed to iterate the set of possible discrete values and n and m are chosen correspondingly.

    Note that this function depends on the experiment design currently set for the model, thus it is crucial that this
    is set to the intended experiment before calling this function.

    Parameters
    ----------
    n: int
        Number of samples of the observation space to use in building the EIG estimate.
    m: int
        Number of samples of the latent variable space to use per observation space sample in building the EIG estimate.
    i_s: compiled callable
        The prior sampling function for the model. When called should generate a sample from the latent variable space.
    y_s: compiled callable
        The proposal observation sampling function for the model. Must generate an observation space sample when called.
    lp_i: compiled callable
        The prior log probability function for the model. Takes a latent variable space sample as input to compute the
        probability of obtaining that sample.
    lp_y: compiled callable
        The likelihood log probability function for the model. Takes a latent variable space sample and observation
        space sample as input to compute the probability of seeing that observation given the latent variable values.

    Returns
    -------
    dict
        A collection of the trace, sampling analysis, and results figures computed
    """

    ### Setup Work ###
    if multicore:
        pool = Pool(processes=4)
    # Provides the capability to test the routine through reproducible sampling
    if multicore and resample_rng:
        raise Exception('Can only set the RNG seed in single core mode')
    elif resample_rng:
        rng = np.random.default_rng() if resample_rng is None else resample_rng
    # Initialize the return dictionary detailing all the results and metrics of the EIG computation
    rtrn_dict = {'sampling_stats': {'smc_resample_keep_percent': np.zeros((ys_per_obs - 1,)),
                                    'low_prob_sample_rate': np.zeros((ys_per_obs,))},
                 'avg_probs': {}, 'issue_symptoms': {}, 'trace': {}, 'results': {}}
    # Initialize variables to track how well-behaved the EIG estimation computation runs
    neg_eig, neg_ig_cnt = 0.0, 0
    # Trace variables
    if compute_traces:
        tr_ig = np.zeros((ys_per_obs, int(n / TR_GAP), int(m / TR_GAP)))
        tr_eig = np.zeros((ys_per_obs, int(n / TR_GAP)))
    # Variables required for computation
    ig_store = np.zeros((n,))
    metric_store = np.zeros((n,))
    marg_store = np.zeros((n,))
    mig, mig_y = 1e5, None
    # Determine the shape of the observation space, define the variables needed for SMC resampling
    y_test = y_s()
    #ys = np.zeros((n, ys_per_obs, len(y_test), len(y_test[0]), len(y_test[0][0])))
    #y_buf = np.zeros((n, ys_per_obs, len(y_test), len(y_test[0]), len(y_test[0][0])))
    ys = np.empty((n, ys_per_obs), dtype=object)
    y_buf = np.empty((n, ys_per_obs), dtype=object)
    zeros = [np.zeros((len(y_test[i]), len(y_test[i][0]))) for i in range(len(y_test))]
    for n_i in range(n):
        for obs in range(ys_per_obs):
            ys[n_i, obs] = zeros.copy()
            y_buf[n_i, obs] = zeros.copy()

    # Determine the shape of the computed lifespans, i.e., how many product lifespans are included in each sample
    i_test = i_s()
    lf_test = f_l(*i_test)
    lf_dims = lf_test.shape

    # Estimate the average magnitude model probabilities, used as constant factors to maintain numeric stability
    lp_i_avg, lp_y_avg = 0.0, 0.0
    if p_y_stabilization or p_i_stabilization:
        lp_i_accum, lp_y_accum = 0.0, 0.0
        for _ in range(m):
            i = i_s()
            y = y_s()
            lp_i_accum += lp_i(*i)
            lp_y_accum += lp_y(*i, *y)
        temp_lp_i_avg = lp_i_accum / m
        temp_lp_y_avg = lp_y_accum / m
        # FIXME: High variance of lp_i or lp_y can make these averages more hindrance than help, need to make smarter
        if p_i_stabilization:
            lp_i_avg = temp_lp_i_avg
            rtrn_dict['avg_probs']['lp_i_avg'] = lp_i_avg
        if p_y_stabilization:
            lp_y_avg = temp_lp_y_avg
            rtrn_dict['avg_probs']['lp_y_avg'] = lp_y_avg
        # TODO: Add warnings and maybe errors for if the average probabilities indicate that the model is poorly built
        if temp_lp_i_avg > 10:
            raise Warning(f"Average latent space log probability of {temp_lp_i_avg} is extremely high...")


    ### Computation time ###
    # Across each partial observation required to produce full observations
    for y_i in range(ys_per_obs):
        print(f"Starting partial observation {y_i} at {datetime.datetime.now()}...")
        outs = []
        to_run = partial(one_obs_process, y_i=y_i, m=m, i_s=i_s, y_s=y_s, lp_i=lp_i, lp_y=lp_y,
                         lp_i_avg=lp_i_avg, lp_y_avg=lp_y_avg, traces=compute_traces,
                         f_l=f_l, f_l_dims=lf_dims, metric_region_size=credible_region_size)
        if multicore:
            with pool:
                unique_args = zip(range(n), ys)
                outs = pool.starmap(to_run, unique_args)
        else:
            for n_i in range(n):
                outs.append(to_run(n_i, ys[n_i]))
        ys = np.array([item[0] for item in outs])
        ig_store = np.array([item[1] for item in outs])
        print(ig_store)
        marg_store = np.array([item[2] for item in outs])
        metric_store = np.array([item[3] for item in outs])

        if compute_traces:
            tr_outs = outs[(TR_GAP - 1)::TR_GAP]
            for tr_i in range(int(n / TR_GAP)):
                tr_ig[y_i, tr_i] = tr_outs[tr_i][4]
                # EIG trace computation using all samples up to each traced point
                end = (tr_i + 1) * TR_GAP
                tr_eig[y_i, tr_i] = np.sum(ig_store[:end] * marg_store[:end]) / np.sum(marg_store[:end])
        # All extremely low probability sample observations will have had their marginal set to 0; count them up now
        rtrn_dict['sampling_stats']['low_prob_sample_rate'][y_i] = np.count_nonzero(marg_store == 0) / n

        # If this is the final partial output cycle there's no need to resample
        if y_i < ys_per_obs - 1:
            # Sum of all p_marg array elements must be 1 for resampling via numpy's choice
            resample_probs = marg_store / np.sum(marg_store)
            # Resample according to the marginal likelihood of each sample
            resampled_inds = rng.choice(n, (n,), p=resample_probs)
            rtrn_dict['sampling_stats']['smc_resample_keep_percent'][y_i] = len(np.unique(resampled_inds)) / n
            for n_i in range(n):
                y_buf[n_i] = ys[resampled_inds[n_i]]
            ys = y_buf

    ### Results Analysis ###
    # Stability checks and output determination
    for n_i in range(n):
        if ig_store[n_i] < mig and ig_store[n_i] != 0.0:
            mig = ig_store[n_i]
            mig_y = ys[n_i]
        if ig_store[n_i] < 0:
            neg_ig_cnt += 1
            neg_eig += ig_store[n_i] * marg_store[n_i]

    eig = np.sum(ig_store * marg_store) / np.sum(marg_store)
    rtrn_dict['results']['eig'] = eig
    rtrn_dict['results']['vig'] = np.sum((ig_store - eig) ** 2) / n
    rtrn_dict['results']['mig'] = mig
    rtrn_dict['results']['mig_y'] = mig_y
    # Can only compute some statistics if a required information gain RIG threshold was provided
    if rig is not None:
        rtrn_dict['results']['rig'] = rig
        pass_inds = ig_store >= rig
        rtrn_dict['results']['rig_pass_prob'] = np.sum(marg_store[pass_inds]) / np.sum(marg_store)
        rtrn_dict['results']['rig_mig_gap'] = rig - mig
        rtrn_dict['results']['rig_eig_gap'] = rig - eig
        fail_inds = ig_store < rig
        if np.any(fail_inds == True):
            eig_fails_only = np.sum(ig_store[fail_inds] * marg_store[fail_inds]) / np.sum(marg_store[fail_inds])
            rtrn_dict['results']['rig_fails_only_eig_gap'] = rig - eig_fails_only
            rtrn_dict['results']['rig_fails_only_vig'] = np.sum((ig_store[fail_inds] - eig_fails_only) ** 2) / len(fail_inds)

    # Can only compute some statistics if a lifespan function is defined
    if f_l is not None:
        expected_metric = np.sum(metric_store * marg_store) / np.sum(marg_store)
        rtrn_dict['results']['e_metric'] = expected_metric
        rtrn_dict['results']['v_metric'] = np.sum((metric_store - expected_metric) ** 2) / n
        min_metric_ind = np.argmin(metric_store)
        rtrn_dict['results']['m_metric'] = metric_store[min_metric_ind]
        rtrn_dict['results']['m_metric_y'] = ys[min_metric_ind]
        if metric_trgt is not None:
            rtrn_dict['results']['metric_trgt'] = metric_trgt
            pass_inds = metric_store >= metric_trgt
            rtrn_dict['results']['metric_pass_prob'] = np.sum(marg_store[pass_inds]) / np.sum(marg_store)
            rtrn_dict['results']['m_metric_trgt_gap'] = metric_trgt - metric_store[min_metric_ind]
            rtrn_dict['results']['e_metric_trgt_gap'] = metric_trgt - expected_metric
            fail_inds = metric_store < metric_trgt
            if np.any(fail_inds == True):
                expected_metric_fails_only = np.sum(metric_store[fail_inds] * marg_store[fail_inds]) / np.sum(marg_store[fail_inds])
                rtrn_dict['results']['metric_fails_only_e_metric_trgt_gap'] = metric_trgt - expected_metric_fails_only
                rtrn_dict['results']['metric_fails_only_v_metric'] = np.sum((metric_store[fail_inds] - expected_metric_fails_only) ** 2) / len(fail_inds)

    rtrn_dict['issue_symptoms']['neg_ig_rate'] = neg_ig_cnt / n
    rtrn_dict['issue_symptoms']['neg_ig_tot'] = neg_eig / np.sum(marg_store)
    if compute_traces:
        rtrn_dict['trace']['ig'] = tr_ig
        rtrn_dict['trace']['eig'] = tr_eig
    return rtrn_dict


def bed_runner(l, n, m, exp_sampler, exp_handle, ltnt_sampler, obs_sampler, logp_prior, logp_likely, rig=0.0,
               life_func=None, life_trgt=None):
    """
    Engine to estimate the optimal experimental design to run for a given model and limited space of possible
    experiments.

    Note that for discrete experiment design spaces the sampler can be a simple iterator that yields each possible
    experiment in turn, with 'l' set accordingly.

    Returns
    -------

    """
    eig_pairs = []
    i_max = None

    # TODO: Sanity checks to avoid attempting BED with bad inputs.
    y_test = obs_sampler()
    obs_len = y_test[0].shape[-1]

    if rig != 0.0:
        rprt_cols = ['design', 'rig_pass_prob', 'rig_eig_gap', 'vig', 'rig_mig_gap',
                     'rig_fails_only_eig_gap', 'rig_fails_only_vig']
    else:
        rprt_cols = ['design', 'eig', 'vig', 'mig']

    for i in range(l):
        d = exp_sampler()

        # Logic handling for tests with sample sizes larger than the size of a single observation
        if type(next(iter(d.values()))) == dict:
            samples_specified = 'samples' in next(iter(d.values())).keys()
            exp_sampler_format = 'per_test'
        else:
            samples_specified = 'samples' in d.keys()
            exp_sampler_format = 'global'
        if samples_specified:
            if exp_sampler_format == 'per_test':
                # POSSIBLE TODO: Currently all experiments in a test need to have the same number of ys_in_exp
                samples_in_exp = next(iter(d.values()))['samples']
                for exp in d:
                    d[exp].pop('samples')
            else:
                samples_in_exp = d['samples']
                d.pop('samples')
            if samples_in_exp % obs_len != 0:
                raise Exception(f"Cannot use experiment measuring {samples_in_exp} devices, must be an integer multiple"
                                f"of the model observation size {obs_len}.")
            ys_in_exp = int(samples_in_exp / obs_len)
        else:
            ys_in_exp = 1

        start_time = t.time()
        results = eig_smc_refined(n, m, ltnt_sampler, obs_sampler, logp_prior, logp_likely, ys_in_exp,
                                  True, False, True, multicore=False, rig=rig, f_l=life_func, metric_trgt=life_trgt)
        # Create the row that will go in the BED report
        rprt_row = [d]
        for col in rprt_cols:
            if col != 'design':
                rprt_row.append(results['results'][col])
        eig_pairs.append(rprt_row)
        print(f"EIG estimation time: {t.time() - start_time} seconds")

        # Save the trace data and results
        Path('bed_data').mkdir(parents=True, exist_ok=True)
        tr_ig = results['trace'].pop('ig')
        tr_eig = results['trace'].pop('eig')
        jnp.save(f"bed_data/exp_{i}_ig_trace.npy", tr_ig)
        jnp.save(f"bed_data/exp_{i}_eig_trace.npy", tr_eig)
        results.pop('trace')
        with open(f"bed_data/exp_{i}_rslts.json", 'w') as f:
            json.dump(results, f, cls=NumpyEncoder)
        # Generate some graphs of the trace curves
        f1, p1 = plt.subplots()
        for tr_i in range(tr_ig.shape[0]):
            for tr_j in range(tr_ig.shape[1]):
                p1.plot(range(tr_ig.shape[2]), tr_ig[tr_i, tr_j])
        p1.set_title(f"exp_{i}_ig")
        f2, p2 = plt.subplots()
        for tr_i in range(tr_eig.shape[0]):
            p2.plot(range(tr_eig.shape[1]), tr_eig[tr_i], label=tr_i)
        p2.legend(loc='lower right')
        p2.set_title(f"exp_{i}_eig")

        f1.savefig(f"bed_data/exp_{i}_ig_trace.png")
        f2.savefig(f"bed_data/exp_{i}_eig_trace.png")
        plt.close(f1)
        plt.close(f2)

    return pd.DataFrame(eig_pairs, columns=rprt_cols)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, jnp.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def u_pp(lf_samples, p_lkly, p_y):
    posterior = p_lkly / jnp.expand_dims(p_y, -1)
    posterior = jnp.where(jnp.isnan(posterior), 0, posterior)
    clean = jnp.where(lf_samples['faulty'], 0, posterior)
    p_clean = jnp.sum(clean)
    return p_clean


def ig(lp_i, lp_lkly, lp_y):
    """
    Information I(i) = -log(p(i)), entropy H(i) = E_i[I(i)] = -SUM_i[p(i)log(p(i))], however since the samples of i are
    distributed according to the prior, H(i) = -SUM_i

    Using importance sampling drawing samples from the prior:
    E_i[I(i)] ~= (1/n)SUM_i[(p(i)/p(i))I(i)] = (1/n)SUM_i[-log(p(i))]

    Parameters
    ----------
    lp_i: log probabilities of latent space samples from the prior
    lp_lkly
    lp_y

    Returns
    -------

    """
    # Elements in lp_i must be based on samples from the prior
    h_pri = -jnp.sum(lp_i, axis=1) / lp_i.shape[1]

    h_pst = -jnp.sum()

    lp_pst = lp_lkly - jnp.expand_dims(lp_y, 1)
    p_pst = jnp.exp(lp_pst)
    lp_pst_norm = logsumexp(lp_pst, axis=1)
    h_pst = -jnp.sum(lp_pst * p_pst, axis=1) / jnp.exp(lp_pst_norm)
    return h_pri - h_pst


def eig(lp_hyl, lp_lkly, lp_y, lp_y_n):
    infgain = ig(lp_hyl, lp_lkly, lp_y)
    return jnp.sum(infgain * jnp.exp(lp_y_n))


def mtrpp(lp_lkly, p_y, field_lifespan, metric_target, quantile):
    lkly_weights = jnp.exp(lp_lkly).flatten()
    lifespans = field_lifespan.copy().flatten()

    metric = wquantiles.quantile(lifespans, lkly_weights, 1 - (quantile / 100))
    pass_inds = metric >= metric_target
    pass_percent = jnp.sum(p_y[pass_inds]) / jnp.sum(p_y)
    return pass_percent


def bed_run(rng_key, l, n, m, exp_sampler, spm, u_funcs=None, field_test=None):
    eus = []
    keys = rand.split(rng_key, l)
    for i in range(l):
        # Get the next proposal experiment design
        d = exp_sampler()
        if u_funcs is None:
            eu = evaluate_design(keys[i], d, n, m, spm, pred_test=field_test)
        else:
            eu = evaluate_design(keys[i], d, n, m, spm, u_funcs, pred_test=field_test)
        eus.append(eu)
    return eus


def evaluate_design(rng_key, d, n, m, spm, u_funcs=eig, pred_test=None):
    """This function evaluates an experimental design utility by sampling from the latent space and prior observation
    space."""
    k1, k2, k3, k4, k5 = rand.split(rng_key, 5)


    # All input names must be standardized names for quantities OR site names of an SPM
    # Note: utility functions can not use other utility function quantities as inputs, instead create wrappers
    utilities = u_funcs if type(u_funcs) == list else [u_funcs]
    ins = []
    for ut in utilities:
        sig = signature(ut)
        for prm in sig.parameters:
            if sig.parameters[prm].default is Parameter.empty:
                ins.append(prm)
    # Remove all duplicates from the list by casting to dictionary then back
    ins = list(dict.fromkeys(ins))

    # TODO: Potentially determine the normalization factors here, though there's a question of whether it's reasonable
    #       to have different normalization factors per experiment design 'd'. It may need to be done in the outer loop to
    #       avoid biasing results
    lp_i_avg, lp_hyl_avg, lp_y_avg = 0.0, 0.0, 0.0
    probs = {}

    # First generate 'n' samples of possible experiment outcomes
    if not spm.y_s_override:
        y_s = spm.sample(k1, d, num_samples=(n,), keep_sites=spm.observes)
    else:
        y_s = spm.y_s_custom(d, num_samples=(n,))
    # Now generate 'm' samples of possible latent space values for each of the 'n' outcomes
    if not spm.i_s_override:
        i_s = spm.sample(k2, d, num_samples=(n, m), keep_sites=spm.ltnt_subsamples + spm.hyls)
    else:
        i_s = spm.i_s_custom(d, num_samples=(n, m))
        probs['lp_s'] = spm.lp_s_custom(d, num_samples=(n, m))
    # The latent space consists of latents and hyper-latents, experiment value only depends on how much is learned about
    # the hyper-latents, so extract them here
    hyl_s = {k: i_s[k] for k in i_s if k in spm.hyls}

    # Now compute log probabilities, subtracting normalization factors to try and bias everything towards probability 1
    # for numerical stability
    probs['lp_i'] = spm.logp(k3, d, i_s, i_s, (n, m)) - lp_i_avg
    probs['lp_hyl'] = spm.logp(k4, d, hyl_s, hyl_s, (n, m)) - lp_hyl_avg
    y_tiled = {y: jnp.repeat(jnp.expand_dims(y_s[y], 1), m, axis=1) for y in y_s}
    probs['lp_ygi'] = spm.logp(k5, d, y_tiled, i_s, (n, m)) - lp_y_avg
    probs['lp_joint'] =spm.logp(k5, d, y_tiled | hyl_s, i_s | y_tiled)

    # Compute probabilities of latent space samples
    lp_i_norm = logsumexp(probs['lp_i'], axis=1, keepdims=True)
    probs['lp_i_n'] = probs['lp_i'] - lp_i_norm
    probs['p_i'] = jnp.exp(probs['lp_i'])
    probs['p_i_n'] = jnp.exp(probs['lp_i_n'])

    # Compute probabilities of hyper-latents
    lp_hyl_norm = logsumexp(probs['lp_hyl'], axis=1, keepdims=True)
    probs['lp_hyl_n'] = probs['lp_hyl'] - lp_hyl_norm
    probs['p_hyl'] = jnp.exp(probs['lp_hyl'])
    probs['p_hyl_n'] = jnp.exp(probs['lp_hyl_n'])

    # Compute likelihoods
    probs['lp_lkly'] = probs['lp_ygi'] + probs['lp_i']
    probs['lp_lkly_ni'] = probs['lp_ygi'] + probs['lp_i'] - lp_i_norm
    lp_lkly_norm = logsumexp(probs['lp_lkly_ni'], axis=1, keepdims=True)
    probs['lp_lkly_n'] = probs['lp_lkly_ni'] - lp_lkly_norm
    probs['p_lkly'] = jnp.exp(probs['lp_lkly'])
    probs['p_lkly_ni'] = jnp.exp(probs['lp_lkly_ni'])
    probs['p_lkly_n'] = jnp.exp(probs['lp_lkly_n'])

    #p_lkly = jnp.where(p_lkly == jnp.nan, 0.0, p_lkly)
    #p_lkly = jnp.where(p_lkly < LOW_PROB_CUTOFF, 0.0, p_lkly)

    # Compute marginal probabilities of observations
    probs['lp_y'] = logsumexp(probs['lp_ygi'], axis=1) - probs['lp_ygi'].shape[1]
    lp_y_norm = logsumexp(probs['lp_y'])
    probs['lp_y_n'] = probs['lp_y'] - lp_y_norm
    probs['p_y'] = jnp.exp(probs['lp_y'])
    probs['p_y_n'] = jnp.exp(probs['lp_y_n'])

    probs['lp_lkly_ny'] = probs['lp_lkly_ni'] - lp_y_norm

    # Generate the predictor variables
    pred_trace = None
    if pred_test:
        # None of the latent space sample sites depends on test conditions, so their values are fixed for the trace
        pred_trace = spm.sample(k5, pred_test, num_samples=(n, m), conditionals=i_s)

    # Compute all the quantities required by the set of utility functions
    utility_ins = {}
    for quantity in ins:
        if quantity in probs:
            utility_ins[quantity] = probs[quantity]
        else:
            if pred_trace is None:
                raise Exception('No predictor test defined, cannot sample predictor values')
            utility_ins[quantity] = pred_trace[quantity]

    # Call the requested utility functions with their required inputs
    u = {}
    for ut in utilities:
        ut_args = {arg: val for arg, val in utility_ins.items() if arg in signature(ut).parameters}
        ut_name = ut.func.__name__ if isinstance(ut, partial) else ut.__name__
        u[ut_name] = ut(**ut_args)

    # Return the computed utility values
    return u


def ig_new(w, w_norm, lp_x, lp_y, lp_y_g_x, e_x):
    lp_x_expanded = jnp.expand_dims(lp_x, axis=(1, 2))
    # NOTE: This is the same as info = -lp_x_expanded - lw
    info = lp_y - lp_x_expanded - lp_y_g_x
    e_x_g_y = jnp.sum(w * info, axis=0) / w_norm
    return e_x - e_x_g_y


def qx_lbci(w, z, q):
    z_ty = jnp.repeat(jnp.expand_dims(z, axis=2), w.shape[2], axis=2)
    w_t = jnp.repeat(w, z.shape[1], axis=1)
    return w_quantile(z_ty, w_t, q)


def eig_new(ig):
    return {'eig': jnp.sum(ig) / ig.size}


def sample_hybrid_utility(ig, qx_lbci, trgt, p_y):
    mtrpp = jnp.sum(jnp.where(qx_lbci > trgt, p_y, 0)) / jnp.sum(p_y)
    return {'eig': jnp.sum(ig) / ig.size, 'qx_lbci': mtrpp}


def reindex(a, i):
    return a[i]


def est_lp_y_g_x(rng_key, spm, d: ReliabilityTest, x_s, y_s, n_v):
    # Figure out the dimensions of the problem and derive random keys from the seed key
    n_x, n_y = next(iter(x_s.values())).shape[0], next(iter(y_s.values())).shape[0]
    n_lot, n_chp = next(iter(y_s.values())).shape[3], next(iter(y_s.values())).shape[2]
    k_init, kdum, kl, kc, kd = rand.split(rng_key, 5)
    perf_stats = {}

    # Tile the x and y sample arrays to match the problem dimensions for full vectorization
    x_s_t = {x: jnp.repeat(jnp.repeat(jnp.expand_dims(x_s[x], (1, 2)), n_v, axis=1), n_y, axis=2) for x in x_s}
    y_s_t = {y: jnp.repeat(jnp.repeat(jnp.expand_dims(y_s[y], axis=(0, 1)), n_v, axis=1), n_x, axis=0) for y in y_s}

    v_init = spm.sample(k_init, d, num_samples=(n_x, n_v, n_y), keep_sites=spm.ltnt_subsamples, conditionals=x_s_t)
    v_dev_zeros = {v: jnp.zeros_like(v_init[v]) for v in v_init if '_dev' in v}
    v_chp_zeros = {v: jnp.zeros_like(v_init[v]) for v in v_init if '_chp' in v}
    v_lot_zeros = {v: jnp.zeros_like(v_init[v]) for v in v_init if '_lot' in v}

    # Evaluate similarity of v to y
    y_approx_zeros = spm.sample(kdum, d, num_samples=(n_x, n_v, n_y), keep_sites=spm.observes, conditionals=x_s_t | v_dev_zeros | v_chp_zeros | v_lot_zeros)

    ##################################
    # First perform lot-level resampling
    ##################################
    lot_ltnts = [ltnt for ltnt in v_init if '_lot' in ltnt]
    v_rs_lot = {}
    if len(lot_ltnts) > 0:
        # Spectacular triple vmap for n_x, n_y, and n_lot dimensions
        choice_vec = jax.vmap(jax.vmap(jax.vmap(rand.choice, (0, 0, None, None, 0), 0), (1, 2, None, None, 2), 2), (2, 3, None, None, 3), 3)
        vec_reindex = jax.vmap(jax.vmap(jax.vmap(reindex, (0, 0), 0), (2, 2), 2), (3, 3), 3)

        for exp in d.config:
            exp_lot_ltnts = [ltnt for ltnt in lot_ltnts if f'{exp}_' in ltnt]
            d_exp = ReliabilityTest({exp: d.config[exp]}, {exp: d.conditions[exp]})
            exp_y_s_t = {y: y_s_t[y] for y in y_s_t if f'{exp}_' in y}

            v_s_lot_init = {ltnt: v_init[ltnt] for ltnt in exp_lot_ltnts}
            # Get the log probabilities without summing so each lot can be considered individually
            lp_y_g_xv = spm.logp(kdum, d_exp, site_vals=exp_y_s_t, conditional=x_s_t | v_s_lot_init | v_dev_zeros | v_chp_zeros, dims=(n_x, n_v, n_y), sum_lps=False)
            # Number of devices might be different for each observed variable, so have to sum across devices and chips
            lp_y_g_xv_lot = {y: jnp.sum(lp_y_g_xv[y], axis=(3, 4)) for y in lp_y_g_xv}
            # Element-wise addition of log-probabilities across different observe variables now that the dimensions match
            lp_y_g_xv_lot_tot = sum(lp_y_g_xv_lot.values())
            # Sum of all p_marg array elements must be 1 for resampling via random choice
            resample_probs = lp_y_g_xv_lot_tot - logsumexp(lp_y_g_xv_lot_tot, axis=1, keepdims=True)
            # Resample according to relative likelihood, need to resample indices so that resamples are the same for each lot-level latent variable
            inds_array = jnp.repeat(jnp.repeat(jnp.repeat(jnp.expand_dims(jnp.arange(n_v), (0, 2, 3)), n_x, axis=0), n_y, axis=2), n_lot, axis=3)
            krs = jnp.reshape(rand.split(kl, n_x * n_y * n_lot), (n_x, n_y, n_lot))
            resample_inds = choice_vec(krs, inds_array, (n_v,), True, jnp.exp(resample_probs))

            v_rs_lot |= {v: vec_reindex(v_s_lot_init[v], resample_inds) for v in exp_lot_ltnts}

        perf_stats['lot_rs_diversity'] = sum([jnp.unique(v_rs_lot[v]).size / (n_x * n_v * n_y * n_lot) for v in lot_ltnts]) / len(lot_ltnts)
        y_approx_lot = spm.sample(kdum, d, num_samples=(n_x, n_v, n_y), keep_sites=spm.observes, conditionals=x_s_t | v_dev_zeros | v_chp_zeros | v_rs_lot)

    ##################################
    # Next up are chip-level variables
    ##################################
    chp_ltnts = [ltnt for ltnt in v_init if '_chp' in ltnt]
    v_rs_chp = {}
    if len(chp_ltnts) > 0:
        choice_vec = jax.vmap(jax.vmap(jax.vmap(jax.vmap(rand.choice, (0, 0, None, None, 0), 0), (1, 2, None, None, 2), 2), (2, 3, None, None, 3), 3), (3, 4, None, None, 4), 4)
        vec_reindex = jax.vmap(jax.vmap(jax.vmap(jax.vmap(reindex, (0, 0), 0), (2, 2), 2), (3, 3), 3), (4, 4), 4)

        for exp in d.config:
            exp_chp_ltnts = [ltnt for ltnt in chp_ltnts if f'{exp}_' in ltnt]
            d_exp = ReliabilityTest({exp: d.config[exp]}, {exp: d.conditions[exp]})
            exp_y_s_t = {y: y_s_t[y] for y in y_s_t if f'{exp}_' in y}

            v_s_chp_init = {ltnt: v_init[ltnt] for ltnt in exp_chp_ltnts}
            # Get the log probabilities without summing so each lot can be considered individually
            lp_y_g_xv = spm.logp(kdum, d_exp, site_vals=exp_y_s_t, conditional=x_s_t | v_rs_lot | v_dev_zeros | v_s_chp_init, dims=(n_x, n_v, n_y), sum_lps=False)
            # Number of devices might be different for each observed variable, so have to sum across devices
            lp_y_g_xv_chp = {y: jnp.sum(lp_y_g_xv[y], axis=3) for y in lp_y_g_xv}
            # Element-wise addition of log-probabilities across different observe variables now that the dimensions match
            lp_y_g_xv_chp_tot = sum(lp_y_g_xv_chp.values())
            # Sum of all p_marg array elements must be 1 for resampling via random choice
            resample_probs = lp_y_g_xv_chp_tot - logsumexp(lp_y_g_xv_chp_tot, axis=1, keepdims=True)
            # Resample according to relative likelihood, need to resample indices so that resamples are the same for each chip-level latent variable
            inds_array = jnp.repeat(jnp.repeat(jnp.repeat(jnp.repeat(jnp.expand_dims(jnp.arange(n_v), (0, 2, 3, 4)), n_x, axis=0), n_y, axis=2), n_chp, axis=3), n_lot, axis=4)
            krs = jnp.reshape(rand.split(kc, n_x * n_y * n_chp * n_lot), (n_x, n_y, n_chp, n_lot))
            resample_inds = choice_vec(krs, inds_array, (n_v,), True, jnp.exp(resample_probs))

            v_rs_chp |= {v: vec_reindex(v_s_chp_init[v], resample_inds) for v in exp_chp_ltnts}

        perf_stats['chp_rs_diversity'] = sum([jnp.unique(v_rs_chp[v]).size / (n_x * n_v * n_y * n_chp * n_lot) for v in chp_ltnts]) / len(chp_ltnts)
        y_approx_chp = spm.sample(kdum, d, num_samples=(n_x, n_v, n_y), keep_sites=spm.observes, conditionals=x_s_t | v_dev_zeros | v_rs_chp | v_rs_lot)

    ##################################
    # Finally are device-level variables
    ##################################
    dev_ltnts = [ltnt for ltnt in v_init if '_dev' in ltnt]
    v_rs_dev = {}
    if len(dev_ltnts) > 0:
        # Spectacular quintuple vmap for n_x, n_y, n_dev, n_chp, and n_lot dimensions
        choice_vec = jax.vmap(jax.vmap(jax.vmap(jax.vmap(jax.vmap(rand.choice, (0, 0, None, None, 0), 0), (1, 2, None, None, 2), 2), (2, 3, None, None, 3), 3), (3, 4, None, None, 4), 4), (4, 5, None, None, 5), 5)
        vec_reindex = jax.vmap(jax.vmap(jax.vmap(jax.vmap(jax.vmap(reindex, (0, 0), 0), (2, 2), 2), (3, 3), 3), (4, 4), 4), (5, 5), 5)

        for exp in d.config:
            exp_dev_ltnts = [ltnt for ltnt in dev_ltnts if f'{exp}_' in ltnt]
            d_exp = ReliabilityTest({exp: d.config[exp]}, {exp: d.conditions[exp]})
            exp_y_s_t = {y: y_s_t[y] for y in y_s_t if f'{exp}_' in y}

            v_s_dev_init = {ltnt: v_init[ltnt] for ltnt in exp_dev_ltnts}
            # Get the log probabilities without summing so each lot can be considered individually
            lp_y_g_xv = spm.logp(kdum, d_exp, site_vals=exp_y_s_t, conditional=x_s_t | v_rs_lot | v_s_dev_init | v_rs_chp, dims=(n_x, n_v, n_y), sum_lps=False)

            resample_inds = {}
            for y in lp_y_g_xv:
                n_dev = lp_y_g_xv[y].shape[3]
                # Sum of all p_marg array elements must be 1 for resampling via random choice
                resample_probs = lp_y_g_xv[y] - logsumexp(lp_y_g_xv[y], axis=1, keepdims=True)
                # Resample according to relative likelihood, need to resample indices so that resamples are the same for each chip-level latent variable
                inds_array = jnp.repeat(jnp.repeat(jnp.repeat(jnp.repeat(jnp.repeat(jnp.expand_dims(jnp.arange(n_v), (0, 2, 3, 4, 5)), n_x, axis=0), n_y, axis=2), n_dev, axis=3), n_chp, axis=4), n_lot, axis=5)
                kd, ky = rand.split(kd)
                krs = jnp.reshape(rand.split(ky, n_x * n_y * n_dev * n_chp * n_lot), (n_x, n_y, n_dev, n_chp, n_lot))
                resample_inds[y] = choice_vec(krs, inds_array, (n_v,), True, jnp.exp(resample_probs))

                v_rs_dev |= {v: vec_reindex(v_s_dev_init[v], resample_inds[y]) for v in exp_dev_ltnts if y in v}

        perf_stats['dev_rs_diversity'] = sum([jnp.unique(v_rs_dev[v]).size / (n_x * n_v * n_y * v_rs_dev[v].shape[3] * n_chp * n_lot) for v in dev_ltnts]) / len(dev_ltnts)
        y_approx = spm.sample(kdum, d, num_samples=(n_x, n_v, n_y), keep_sites=spm.observes, conditionals=x_s_t | v_rs_lot | v_rs_chp | v_rs_dev)

    # Final logp
    lp_v_g_x = spm.logp(kdum, d, site_vals=v_rs_dev | v_rs_chp | v_rs_lot, conditional=x_s_t, dims=(n_x, n_v, n_y))
    lp_y_g_xv = spm.logp(kdum, d, site_vals=y_s_t, conditional=x_s_t | v_rs_lot | v_rs_chp | v_rs_dev, dims=(n_x, n_v, n_y))
    lp_y_g_x = logsumexp(lp_v_g_x + lp_y_g_xv, axis=1)
    return lp_y_g_x, perf_stats


def run_bed_newest(rng_key, n_d, n_y, n_v, n_x, d_sampler, spm, utility=eig_new, d_field=None, trgt_lifespan=None):
    k, kd, kx = rand.split(rng_key, 3)
    # Get the first proposal experiment design, need to sample here for dummy input to x_s sample and lp_x logp
    d = d_sampler(kd)
    # We can sample all x~p(x) here since x is independent of the test d, the observations y, and the predictors z
    x_s = spm.sample(kx, d, num_samples=(n_x,), keep_sites=spm.hyls)
    # Compute the prior entropy once, but only if it's needed for EIG estimation
    if 'ig' in signature(utility).parameters:
        lp_x = spm.logp(kx, d, site_vals=x_s, conditional=None, dims=(n_x,))
        e_x = jnp.sum(-lp_x) / n_x

    # We loop over sample test designs instead of vectorizing since test designs can have differing dimensionality
    us = []
    for _ in range(n_d):
        k, kv, ky, ku, kz, kd = rand.split(k, 6)
        # Sample observations from joint prior y~p(x,v,y|d), keeping y independent of the already sampled x and v values
        y_s = spm.sample(ky, d, num_samples=(n_y,), keep_sites=spm.observes)

        # SMC resampling of 'v' samples should occur here
        x_s_tv = {x: jnp.repeat(jnp.expand_dims(x_s[x], axis=1), n_v, axis=1) for x in x_s}
        v_s = spm.sample(kv, d, num_samples=(n_x, n_v), keep_sites=spm.ltnt_subsamples, conditionals=x_s_tv)

        # Compute posterior importance weights
        # Tile x, v, and y to match in size along their 3 dimensions
        y_s_txv = {y: jnp.repeat(jnp.repeat(jnp.expand_dims(y_s[y], axis=(0, 1)), n_x, axis=0), n_v, axis=1) for y in y_s}
        x_s_tvy = {x: jnp.repeat(jnp.expand_dims(x_s_tv[x], axis=2), n_y, axis=2) for x in x_s_tv}
        v_s_ty = {v: jnp.repeat(jnp.expand_dims(v_s[v], axis=2), n_y, axis=2) for v in v_s}
        lp_y_g_xv = spm.logp(ky, d, site_vals=y_s_txv, conditional=x_s_tvy | v_s_ty, dims=(n_x, n_v, n_y))
        # Marginalize across aleatoric uncertainty axis 'v'
        lp_y_g_x = logsumexp(lp_y_g_xv, axis=1, keepdims=True) - jnp.log(n_v)

        # Marginalize across epistemic uncertainty axis 'x'
        lp_y = logsumexp(lp_y_g_x, axis=0, keepdims=True) - jnp.log(n_x)
        # Now can compute the importance weights
        lw_z = lp_y_g_x - lp_y
        lw_z_norm = logsumexp(lw_z, axis=0)
        # TODO: w_z_norm should always come out to a value of n_x just by the properties of algebra used to compute w_z
        w_z, w_z_norm = jnp.exp(lw_z), jnp.exp(lw_z_norm)

        # Sample predictors if needed
        if 'qx_lbci' in signature(utility).parameters:
            z_s = spm.sample(kz, d_field, (n_x, n_v), keep_sites=spm.predictors, conditionals=x_s_tv)

        # Now compute summary statistics for each sample 'y'
        metrics = {}
        for m in signature(utility).parameters:
            # Start by supporting IG and QX%-LBCI metrics, ideally QX%-HDCR too
            match m:
                case 'ig':
                    metrics[m] = ig_new(w_z, w_z_norm, lp_x, lp_y, lp_y_g_x, e_x)
                case 'qx_lbci':
                    metrics[m] = qx_lbci(w_z, z_s['field_z'], 0.01)
                case 'p_y':
                    metrics[m] = jnp.exp(lp_y)
                case 'trgt':
                    metrics[m] = trgt_lifespan
                case _:
                    raise Exception('Unsupported metric requested')

        # Now compute the utility summary statistic for 'd' based on the metrics for each 'y' sample
        u = utility(**metrics)
        us.append({'design': d, 'utility': u})

        # Now update the test design for the next iteration, the final design (index n_d) is not used
        d = d_sampler(kd)
    return us
