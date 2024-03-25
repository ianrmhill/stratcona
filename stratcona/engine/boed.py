# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import warnings
import numpy as np
import pandas as pd
from scipy.special import logsumexp

from stratcona.engine.inference import fit_latent_params_to_posterior_samples

__all__ = ['eig_importance_sampled', 'bed_runner']

NATS_TO_BITS = np.log2(np.e)
# Machine precision is defaulted to 32 bits since most instruments don't have 64 bit precision and/or noise floors. Less
# precise but smaller entropy values makes things nicer to deal with.
CARDINALITY_MANTISSA_32BIT = 2 ** 24


def p_mx(x: np.ndarray, a: np.ndarray, b: np.ndarray):
    """The probability distribution function of a continuous uniform distribution on the interval [a, b]."""
    return np.where(np.less_equal(a, x) & np.less_equal(x, b), 1 / (b - a), 0)


def entropy(x_sampler, p, m=1e5, p_args=None, in_bits=False,
            limiting_density=False, precision=CARDINALITY_MANTISSA_32BIT):
    a, b = [0], [1]
    h = 0.0
    # Entropy is predicated on iterating over all possible values that theta can take, thus the overall probability
    # summed across all samples of theta is 1. In the continuous case or to compute via sampling, we need to normalize
    # with respect to the probability sum across all the samples of theta.
    p_x_theory, h_norm = 0.0, 0.0
    # Metric for detecting whether the full high probability region of the distribution was sampled well
    p_u_theory, p_u_actual = 0.0, 0.0
    for _ in range(int(m)):
        x = x_sampler()
        logp_x = p(*x, *p_args) if p_args else p(*x)

        p_x = np.exp(logp_x)
        p_u = p_mx(*x, *a, *b)
        p_x_theory += p_x
        p_u_theory += p_u

        if limiting_density:
            if np.all(p_u == 0):
                continue
            if p_x > 0:
                prob = p_x / p_u
                logp_x = np.log(prob)
        p_u_actual += p_u
        h -= p_x * logp_x
        h_norm += p_x
    # Support representing entropy in units of 'nats' or 'bits'
    h = h / h_norm

    # Try to inform the user on how well the samples captured the probability distribution
    x_to_u_ratio = h_norm / p_u_actual
    if x_to_u_ratio > 1:
        print(f"Sampling successfully found high probability regions.")
        if x_to_u_ratio > 10:
            print(f"Bounded region of possibility may be unnecessarily large. Ratio: {x_to_u_ratio}.")
    elif x_to_u_ratio < 0.1:
        print(f"Sampling did not find high probability regions. Rework likely required. Ratio: {x_to_u_ratio}.")
    else:
        print(f"Sampling went acceptably well.")

    # Inform the user if the bounded region u may be too small
    oob_percent = (p_x_theory - h_norm) / p_x_theory
    if oob_percent > 0.01:
        print(f"{round(oob_percent * 100, 2)}% of the sampled PDF fell outside of u bounds. You may want to extend u.")
    if limiting_density:
        # For limiting density, the entropy is log(N) + h, where N is the number of discrete points in the interval
        # [a, b]. For our floating point representations, we assume that the exponent bits determine a and b, then
        # N is the number of possible values for the mantissa bits, i.e., the cardinality. This way N doesn't change
        # regardless of if our random variable support is [0, 1] or [-1e10, 1e10] and represents a physical property
        # of our measurements in terms of their precision once converted to floating point representations.
        # Note that 'h' will always be negative in the limiting density formulation.
        h += np.log(precision)
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


def weighted_sample(vals, lp_likelihoods, num_samples, ltnt_names):
    # Normalize the probabilities then generate samples using the indices
    lp_normed = lp_likelihoods - logsumexp(lp_likelihoods)
    sampled_inds = np.random.choice(len(vals), p=np.exp(lp_normed), size=num_samples)

    # Now with the sampled indices, construct a dictionary of sampled values for the variables
    sampled = {ltnt: np.zeros((num_samples, vals.shape[-1])) for ltnt in ltnt_names}
    for i, sample in enumerate(sampled_inds):
        for j, ltnt in enumerate(ltnt_names):
            sampled[ltnt][i] = vals[sample, j]
    return sampled


def eig_importance_sampled(n: int, m: int, i_s, y_s, lp_i, lp_y, ys_per_obs: int = 1,
                           prior_handle=None, ltnt_info=None, raw_eig_norming=True):
    """
    Core EIG (expected information gain) computation function that produces an estimate using sampling. This sampler
    uses importance sampling and has the benefit of being exact for finite discrete models if the samplers i_s and y_s
    are constructed to iterate the set of possible discrete values and n and m are chosen correspondingly. This sampler
    has the drawback of being very slow, as it does not try to take any estimation shortcuts or assume any knowledge
    about the model structure.

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
    float
        The estimated EIG in nats.
    """
    # Initialize variables to track how well-behaved the EIG estimation computation runs
    # TODO: Statistic showing how unevenly the information gain is distributed. If only a few sampled y values account
    #       for a majority of the total EIG than there's likely something that needs changing
    unobs_cnt, high_p_y_cnt, neg_ig, teensy_marg_cnt = 0, 0, 0.0, 0
    # Initialize the accumulator variables that sum up the total information gain and the normalization constant
    mig, mig_y, eig, eig_norm = 1e5, None, 0.0, 0.0

    # We will use up some memory to slightly improve performance
    i_test = i_s()
    i_store = np.zeros((m, len(i_test), len(i_test[0])))
    lp_cond_store = np.zeros((m,))
    lp_pri_store = np.zeros((m,))

    # Estimate the average magnitude of conditional probabilities, used as multiplying constant to maintain stability
    p_y_avg = 0
    for _ in range(m):
        i = i_s()
        y = y_s()
        p_y_avg += np.exp(lp_y(*i, *y))
    p_y_avg /= m

    # Initial compatibility checks for sequential inference based estimation
    seq_inf_est, orig_priors = False, None
    if ys_per_obs > 1:
        seq_inf_est = True
        if prior_handle is None or ltnt_info is None:
            raise Exception('Handle to set priors and distribution info required to leverage sequential inference estimation techniques.')
        else:
            orig_priors = prior_handle.get_params(for_user=True)

    # Across 'n' observation space samples
    for n_i in range(n):
        ig_tot, p_marg, p_marg_tot, obs_y = 0.0, 0.0, 1.0, []
        # Sequential inference estimation loop, only runs once unless multiple y samples are needed for one observation
        int_priors = orig_priors
        for _ in range(ys_per_obs):
            if seq_inf_est:
                prior_handle.set_params(int_priors)
            y = y_s()
            obs_y.append(y)

            # Compute the information gain for the current sample
            for m_i in range(m):
                i = i_s()
                i_store[m_i] = i

                lp_pri = lp_i(*i)
                lp_cond = lp_y(*i, *y)
                # Persist the log prior for later prior entropy and normalization factor computation
                lp_pri_store[m_i] = lp_pri
                # Persist the log importance weighted likelihood
                lp_cond_store[m_i] = lp_cond

            # Batch convert all the stored log probabilities to probability
            p_pri = np.exp(lp_pri_store)
            lp_likely = lp_cond_store + lp_pri_store
            p_likely = np.exp(lp_likely)

            # Summations used to determine the marginal: p(y|d)
            marg = np.sum(p_likely)
            norm = np.sum(p_pri)
            # NOTE: We do not check for norm == 0 as this indicates a problem with compatibility between i_s and lp_i
            # that should already be validated before calling this function
            p_marg = marg / norm
            # If the marginal probability is extremely low this indicates that the current sample y is incredibly unlikely
            # within the prior model
            p_marg_tot *= p_marg if raw_eig_norming else p_marg * p_y_avg
            # TODO: Tune this cutoff for ignoring samples from y_s
            if p_marg < 1e-50:
                # In this case the information gain computation will be extremely unstable, thus we note and skip this 'y'
                unobs_cnt += 1
                continue

            # Compute the un-normalized prior entropy
            n_h_pri = np.sum(p_pri * lp_pri_store)
            # Compute the un-normalized posterior entropy
            lp_post = lp_likely - np.log(p_marg)
            p_post = p_likely / p_marg
            n_h_post = np.sum(p_post * lp_post)
            # Finally, compute the normalized information gain for the sampled 'y'
            ig = (n_h_post - n_h_pri) / norm

            # Check for conditions indicating that lp_y > 0 (p_y > 0) for some sampled 'i' values for the current 'y'. These
            # should normally be compensated for via the p_marg normalization within Bayes' theorem, but since we are using
            # sampling-based estimation it is possible to sample more high-probability region in the second loop
            if ig < 0 or p_marg > 1:
                high_p_y_cnt += 1
                neg_ig += ig * p_marg

            # Summation over sequential inference steps to get the total information gain of the observation
            ig_tot += ig
            # Inference to get new priors for the next sequential inference step
            if seq_inf_est:
                samples = weighted_sample(i_store, lp_cond_store, n, prior_handle.map.keys())
                int_priors = fit_latent_params_to_posterior_samples(ltnt_info, prior_handle.map, samples)

        # Now add the information gain to the experiment design optimization parameters
        if p_marg > 1e-50:
            if ig_tot < mig:
                mig = ig_tot
                mig_y = obs_y
            eig += ig_tot * p_marg_tot
            eig_norm += p_marg_tot
        else:
            teensy_marg_cnt += 1

    # Reset the prior handle to avoid side effects to the model being evaluated
    if seq_inf_est:
        prior_handle.set_params(orig_priors)

    # TODO: Check the issue tracking variables to ensure the computation was well-behaved
    if high_p_y_cnt > 0:
        print(f"Greater than 1 probabilities occurring in {(high_p_y_cnt / n) * 100}% of sample observations.")
    if unobs_cnt > 0:
        print(f"{(unobs_cnt / (n * ys_per_obs)) * 100}% of sampled 'y' values were extremely unlikely.")
    print(f"{(teensy_marg_cnt / n) * 100}% of observations were extremely unlikely.")
    # Normalize the estimate for EIG(d) and return it
    return eig / eig_norm


def bed_runner(l, n, m, exp_sampler, exp_handle, ltnt_sampler, obs_sampler, logp_prior, logp_likely,
                return_best_only: bool = False, prior_handle=None, ltnt_info=None):
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
    # TODO: Sanity check that ltnt_sampler is producing samples that aren't impossible based on logp_prior
    y_test = obs_sampler()
    obs_len = y_test[0].shape[-1]

    for i in range(l):
        d = exp_sampler()
        # Logic handling for tests with sample sizes larger than the size of a single observation
        if 'samples' in d.keys():
            samples_in_exp = d['samples']
            if samples_in_exp % obs_len != 0:
                raise Exception(f"Cannot use experiment measuring {samples_in_exp} devices, must be an integer multiple"
                                f"of the model observation size {obs_len}.")
            ys_in_exp = int(samples_in_exp / obs_len)
            d.pop('samples')
        else:
            ys_in_exp = 1
        # Crucial to set the experimental design within the model first, as otherwise the EIG won't correspond to 'd'
        exp_handle.set_experimental_params(d)

        # Now we can estimate EIG(d) and add it to the list
        # TODO: Split into two methods, one normal and one using sequential inference. Currently the method is getting
        #       too complex when they're put together
        eig = eig_importance_sampled(n, m, ltnt_sampler, obs_sampler, logp_prior, logp_likely,
                                     ys_per_obs=ys_in_exp, prior_handle=prior_handle, ltnt_info=ltnt_info,
                                     raw_eig_norming=False)
        eig_pairs.append([d, eig])
        print(f"Exp {i}: {eig} nats")
        # Track which is the best experiment from an expected information gain perspective
        if i_max is None or eig > eig_pairs[i_max][1]:
            i_max = i

    # Return either just the best design found or a full dataframe reporting the designs and EIGs
    return eig_pairs[i_max] if return_best_only else pd.DataFrame(eig_pairs, columns=['design', 'eig'])
