# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd

__all__ = ['eig_sampled', 'boed_runner']

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

# FIXME: FLAWED! Incorrect math occurred when converting to importance sampling approach
def eig_sampled(n: int, m: int, theta_sampler, y_sampler, p_prior, p_likely):
    """
    Core EIG (expected information gain) computation function that produces an estimate using sampling. This estimation
    requires four compiled functions, three of which are generated directly from the problem model, and then a proposal
    sampler for the observation distribution which should approximate the range and likelihood of possible observations.

    Note that this function depends on the experiment design currently set for the model, thus it is crucial that this
    is set to the intended experiment before calling this function.

    Additionally, for discrete models, if the samplers are iterators for countable discrete distributions and n and m
    are set accordingly, the EIG computation will be exact instead of an estimate.

    Parameters
    ----------
    n: int
        Number of samples of the observation space to use in building the EIG estimate.
    m: int
        Number of samples of the latent variable space to use per observation space sample in building the EIG estimate.
    theta_sampler: compiled callable
        The prior sampling function for the model. When called should generate a sample from the latent variable space.
    y_sampler: compiled callable
        The proposal observation sampling function for the model. Must generate an observation space sample when called.
    p_prior: compiled callable
        The prior log probability function for the model. Takes a latent variable space sample as input to compute the
        probability of obtaining that sample.
    p_likely: compiled callable
        The likelihood log probability function for the model. Takes a latent variable space sample and observation
        space sample as input to compute the probability of seeing that observation given the latent variable values.

    Returns
    -------
    float
        The estimated EIG in nats.
    """
    # Initialize the accumulator variables that sum up the total information gain and the normalization constant
    eig, eig_norm = 0, 0
    # Across 'n' observation space samples
    for n_i in range(n):
        y = y_sampler()
        # Initialize accumulators for information gain and marginal probability and associated normalization constants
        ig, ig_norm, marg, marg_norm = 0, 0, 0, 0
        # Across 'm' latent variable space samples
        for m_i in range(m):
            i = theta_sampler()
            # Compute the probabilities for the latent variable space sample [p(theta)] and observation given the latent
            # variable space sample [p(y|theta)], noting that this is implicitly p(y|theta, d) where d is the experiment
            p_i = np.exp(p_prior(*i))
            p_cond = np.exp(p_likely(*i, *y))
            # Contribute to the estimate for the marginal probability p(y|d)
            marg += p_cond * p_i
            marg_norm += p_i
            # As long as the likelihood is non-zero, contribute to the estimate for the information gain IG(y, d)
            if p_cond > 0:
                #print(f"Inner {m_i}, i: {i}, obs prob given sample: {p_cond}, sample prob: {p_i}.")
                ig += np.log(p_cond) * p_i
                ig_norm += p_i

        # Normalize the estimates for the information gain IG(y, d) and the marginal probability p(y|d)
        #print(f"Iter {n_i}, y: {y}, marg sum: {marg}, marg norm: {marg_norm}, IG pre marg: {ig}.")
        marg = marg / marg_norm if marg_norm > 0 else 0
        if marg > 0:
            ig -= np.log(marg) * ig_norm
        ig = ig / ig_norm if ig_norm > 0 else 0
        # With completed estimates for IG(y, d) and p(y|d) we can contribute to the estimate for EIG(d)
        eig += ig * marg
        eig_norm += marg
        #print(f"Iter {n_i}, marg: {marg}, IG: {ig}, IG norm: {ig_norm}.")

    # Normalize the estimate for EIG(d) and return it
    #print(f"EIG norm: {eig_norm}.")
    return eig / eig_norm if eig_norm > 0 else 0.0


# TODO: Seems to be working, fixes the problem with eig_sampled! Now rearrange and optimize to be as fast as possible
def eig_sampled_unoptimized(n: int, m: int, theta_sampler, y_sampler, p_prior, p_likely):
    # Initialize the accumulator variables that sum up the total information gain and the normalization constant
    eig, eig_norm = 0.0, 0.0
    # Across 'n' observation space samples
    for n_i in range(n):
        y = y_sampler()
        # Compute the marginal
        marg, marg_norm = 0.0, 0.0
        for m_i in range(m):
            i = theta_sampler()
            p_i = np.exp(p_prior(*i))
            p_cond = np.exp(p_likely(*i, *y))

            # Marginal computation
            marg += p_cond * p_i
            marg_norm += p_i

        # If the marginal is zero, this y sample doesn't contribute to the overall EIG estimate. Term 'marg_norm' will
        # only ever be zero if 'marg' is also zero
        if marg == 0:
            continue
        marginal = marg / marg_norm

        pri, post, ig_norm = 0.0, 0.0, 0.0
        for m_i in range(m):
            i = theta_sampler()
            p_i = np.exp(p_prior(*i))
            if p_i > 0:
                p_cond = np.exp(p_likely(*i, *y))
                # When p_cond is zero, we use the fact that the limit as p->0 of p * log(p) is 0
                p_post = (p_cond * p_i) / marginal
                if p_post > 0:
                    post += np.log(p_post) * p_post
                pri += np.log(p_i) * p_i
                ig_norm += p_i
                    #print(f"Inner {m_i} - i: {i}, pri: {p_i}, post: {p_post}.")

        h_pri, h_post = pri / ig_norm, post / ig_norm
        ig = h_post - h_pri # Signs flipped since we never added the negatives to the entropy
        #print(f"Iter {n_i} - y: {y}, marginal: {marginal}, prior entropy: {-h_pri}, post entropy: {-h_post}, IG: {ig}.")
        eig += ig * marginal
        eig_norm += marginal

    # Normalize the estimate for EIG(d) and return it
    print(f"EIG norm: {eig_norm}.")
    return eig / eig_norm


def boed_runner(l, n, m, exp_sampler, exp_handle, ltnt_sampler, obs_sampler, logp_prior, logp_likely,
                return_best_only: bool = False):
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

    for i in range(l):
        d = exp_sampler()
        # Crucial to set the experimental design within the model first, as otherwise the EIG won't correspond to 'd'
        exp_handle.set_experimental_params(d)
        # Now we can estimate EIG(d) and add it to the list
        #eig = eig_sampled(n, m, ltnt_sampler, obs_sampler, logp_prior, logp_likely)
        eig = eig_sampled_unoptimized(n, m, ltnt_sampler, obs_sampler, logp_prior, logp_likely)
        eig_pairs.append([d, eig])
        print(f"Exp {i}: {eig} nats")
        # Track which is the best experiment from an expected information gain perspective
        if i_max is None or eig > eig_pairs[i_max][1]:
            i_max = i

    # Return either just the best design found or a full dataframe reporting the designs and EIGs
    return eig_pairs[i_max] if return_best_only else pd.DataFrame(eig_pairs, columns=['design', 'eig'])
