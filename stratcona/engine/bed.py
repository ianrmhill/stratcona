# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import time as t
import datetime
import json
import warnings
import jax.numpy as jnp
import jax.random as rand
import wquantiles
import pandas as pd

from multiprocessing import Pool
from functools import partial
from typing import Callable

from matplotlib import pyplot as plt


__all__ = ['eig_smc_refined', 'bed_runner']

NATS_TO_BITS = jnp.log2(jnp.e)
# Machine precision is defaulted to 32 bits since most instruments don't have 64 bit precision and/or noise floors. Less
# precise but smaller entropy values makes things nicer to deal with.
CARDINALITY_MANTISSA_64BIT = float(2 ** 53)
# Marginal probability below which a sample observed 'y' will be excluded from EIG computation
LOW_PROB_CUTOFF = 1e-20
# Gap between trace samples to reduce memory and performance impact
TR_GAP = 20


def p_mx(x: jnp.ndarray, a: float, b: float):
    """The probability distribution function of a continuous uniform distribution on the interval [a, b]."""
    return jnp.average(jnp.where(jnp.less_equal(a, x) & jnp.less_equal(x, b), 1 / (b - a), 0))


def entropy(samples, lp_func, p_args=None, in_bits=False,
            limiting_density_range: tuple = None, precision=CARDINALITY_MANTISSA_64BIT):
    # Entropy is predicated on iterating over all possible values that theta can take, thus the overall probability
    # summed across all samples of theta is 1. In the continuous case or to compute via sampling, we need to normalize
    # with respect to the probability sum across all the samples of theta.
    lps = lp_func(samples, *p_args) if p_args else lp_func(samples)
    p_s = jnp.exp(lps)
    p_s_theory = jnp.sum(p_s)

    if limiting_density_range is not None:
        a, b = limiting_density_range[0], limiting_density_range[1]
        p_u = p_mx(samples, a, b)
        p_u_tot = jnp.sum(p_u)
        prob = p_s / p_u
        lps = jnp.log(prob)

    h = jnp.sum(-p_s * lps)
    h_norm = jnp.sum(p_s)
    h = h / h_norm

    # Try to inform the user on how well the samples captured the probability distribution
    # Metric for detecting whether the full high probability region of the distribution was sampled well
    if limiting_density_range is not None:
        s_to_u_ratio = h_norm / p_u_tot
        if s_to_u_ratio > 1:
            print(f"Sampling successfully found high probability regions.")
            if s_to_u_ratio > 10:
                print(f"Bounded region of possibility may be unnecessarily large. Ratio: {s_to_u_ratio}.")
        elif s_to_u_ratio < 0.1:
            print(f"Sampling did not find high probability regions. Rework likely required. Ratio: {s_to_u_ratio}.")
        else:
            print(f"Sampling went acceptably well.")

        # Inform the user if the bounded region u may be too small
        oob_percent = (p_s_theory - h_norm) / p_s_theory
        if oob_percent > 0.01:
            print(f"{round(oob_percent * 100, 2)}% of the sampled PDF fell outside of u bounds. You may want to extend u.")

        # For limiting density, the entropy is log(N) + h, where N is the number of discrete points in the interval
        # [a, b]. For our floating point representations, we assume that the exponent bits determine a and b, then
        # N is the number of possible values for the mantissa bits, i.e., the cardinality. This way N doesn't change
        # regardless of if our random variable support is [0, 1] or [-1e10, 1e10] and represents a physical property
        # of our measurements in terms of their precision once converted to floating point representations.
        # Note that 'h' will always be negative until the precision has been added in the limiting density formulation.
        h += jnp.log(precision)

    # Support representing entropy in units of 'nats' or 'bits'
    return h if not in_bits else h * NATS_TO_BITS


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
            lp_post = lp_lkly_store[:end] - np.log(marg)
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


def u_ig(lp_i, p_i, lp_lkly, p_y):
    lp_post = lp_lkly - jnp.expand_dims(jnp.where(p_y > 0, jnp.log(p_y), jnp.inf), 1)
    p_post = jnp.exp(lp_post)

    h_pri = -jnp.sum(lp_i * p_i, axis=1)
    h_post = -jnp.sum(lp_post * p_post, axis=1)

    return (h_pri - h_post) / jnp.sum(p_i, axis=1)


def u_pp(lf_samples, p_lkly, p_y):
    posterior = p_lkly / jnp.expand_dims(p_y, -1)
    posterior = jnp.where(jnp.isnan(posterior), 0, posterior)
    clean = jnp.where(lf_samples['faulty'], 0, posterior)
    p_clean = jnp.sum(clean)
    return p_clean


def eig(lp_i, p_i, lp_lkly, p_y):
    p_y_norm = jnp.sum(p_y)
    ig = u_ig(lp_i, p_i, lp_lkly, p_y)
    return jnp.sum(ig * p_y) / p_y_norm



def mtrpp(lp_lkly, p_y, lifespan, metric_target, quantile):
    lkly_weights = jnp.exp(lp_lkly)
    lifespans = lifespan.copy()
    if len(f_l_dims) > 0:
        lkly_weights = jnp.repeat(lkly_weights, int(lifespan.size / lkly_weights.size))
        lifespans = lifespans.flatten()

    metric = wquantiles.quantile(lifespans, lkly_weights, 1 - (quantile / 100))
    pass_inds = metric >= metric_target
    pass_percent = jnp.sum(p_y[pass_inds]) / jnp.sum(p_y)
    return pass_percent


def bed_run(rng_key, l, n, m, exp_sampler, spm, u_funcs=None):
    eus = []
    keys = rand.split(rng_key, l)
    for i in range(l):
        # Get the next proposal experiment design
        d = exp_sampler()
        if u_funcs is None:
            eu = evaluate_design(keys[i], d, n, m, spm)
        else:
            eu = evaluate_design(keys[i], d, n, m, spm, u_funcs)
        eus.append(eu)
        print(f'{d.conditions}: {eu * NATS_TO_BITS} bits')
        #print(f'{d.conditions}: {eu * 100}%')
    return eus


def evaluate_design(rng_key, d, n, m, spm, u_funcs=eig):
    k1, k2, k3, k4, k5 = rand.split(rng_key, 5)
    # TODO: Potentially determine the normalization factors here, though there's a question of whether it's reasonable
    #       to have different normalization factors per experiment design 'd'. It may need to be done in the outer loop to
    #       avoid biasing results
    lp_i_avg, lp_hyl_avg, lp_y_avg = 0.0, 0.0, 0.0

    # First generate 'n' samples of possible experiment outcomes
    if not spm.y_s_override:
        y_s = spm.sample(k1, d, num_samples=(n,), keep_sites=spm.test_measurements)
    else:
        y_s = spm.y_s_custom(d, num_samples=(n,))
    # Now generate 'm' samples of possible latent space values for each of the 'n' outcomes
    if not spm.i_s_override:
        i_s = spm.sample(k2, d, num_samples=(n, m), keep_sites=spm.ltnts + spm.hyls)
    else:
        i_s = spm.i_s_custom(d, num_samples=(n, m))
    # The latent space consists of latents and hyper-latents, experiment value only depends on how much is learned about
    # the hyper-latents, so extract them here
    hyl_s = {k: i_s[k] for k in i_s if k in spm.hyls}

    # Now compute log probabilities, subtracting normalization factors to try and bias everything towards probability 1
    # for numerical stability
    lp_i = spm.logp(k3, d, i_s, i_s, (n, m)) - lp_i_avg # TODO: Check exactly how log_prob of a variable works in relation to conditioning
    lp_hyl = spm.logp(k4, d, hyl_s, None, (n, m)) - lp_hyl_avg
    y_tiled = {y: jnp.repeat(jnp.expand_dims(y_s[y], 1), m, axis=1) for y in y_s}
    lp_ygi = spm.logp(k5, d, y_tiled, i_s, (n, m)) - lp_y_avg

    # Compute likelihood weightings and the marginal probability of each sampled observation
    p_i = jnp.exp(lp_i)
    i_norm = jnp.sum(p_i, axis=1)

    lp_lkly = lp_ygi + lp_i
    p_lkly = jnp.exp(lp_lkly)
    p_lkly = jnp.where(p_lkly == jnp.nan, 0.0, p_lkly)
    p_lkly = jnp.where(p_lkly < LOW_PROB_CUTOFF, 0.0, p_lkly)

    p_y = jnp.sum(p_lkly, axis=1) / i_norm
    p_y_norm = jnp.sum(p_y)
    p_y_normed = p_y / p_y_norm

    # Compute other necessary inputs for the utility functions
    # All input names must be standardized names for quantities OR site names of an SPM
    utility_ins = {lp_i.__name__: lp_i, p_i.__name__: p_i, lp_lkly.__name__: lp_lkly, p_y.__name__: p_y}

    # Call the requested utility functions with their required inputs
    u = {}
    utilities = u_funcs if type(u_funcs) == list else [u_funcs]
    for func in utilities:
        u[func.__name__] = func(**utility_ins)

    # Assemble a report or just return the final utility
    return u

    # Compute utility measures for individual sample observations
    # TODO: Switching logic for different and multiple utility definitions
    #u = u_ig(lp_i, p_i, lp_lkly, p_y)

    #lf_samples = spm.sample_new(k5, d, num_samples=(n, m), keep_sites=spm.life_predictors, conditionals=i_s)
    #u = u_pp(lf_samples, p_lkly, p_y)
    #return u
    # Zero out the utility of extremely low probability observation space samples, as these often end up with infinite
    # utility due to numerical instability of log and division operations
    #u = jnp.where(p_y < LOW_PROB_CUTOFF, 0.0, u)

    # Compute metrics across all sampled observations
    e_u = jnp.sum(u * p_y) / p_y_norm
    v_u = jnp.sum((u - e_u) ** 2) / n
    m_ind = int(jnp.argmin(u))
    m_u = u[m_ind]
    m_y = {obs: y_s[obs][m_ind] for obs in y_s}
    # Some metrics require a reference threshold value to compute
    if u_threshold is not None:
        pass_inds = u >= u_threshold
        pp_u = jnp.sum(p_y[pass_inds]) / p_y_norm
        mg_u = u_threshold - u[m_ind]
        eg_u = u_threshold - e_u
        fail_inds = u < u_threshold
        # Some metrics only make sense if at least one sampled observation failed to meet the reference threshold
        if jnp.any(fail_inds == True):
            ef_u = jnp.sum(u[fail_inds] * p_y[fail_inds]) / jnp.sum(p_y[fail_inds])
            efg_u = u_threshold - ef_u
            efv_u = jnp.sum((u[fail_inds] - ef_u) ** 2) / len(fail_inds)

    # Support representing entropy in units of 'nats' or 'bits'
    return e_u if not entropy_in_bits else e_u * NATS_TO_BITS
