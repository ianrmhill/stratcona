# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd

__all__ = ['eig_sampled', 'boed_runner']


def ig_sampled(m: int, theta_sampler, p_prior, p_likely):
    """
    Placeholder for now, may be nice to have in the future for comparing the actual information gain to the expected
    gain after running experiments.
    """
    pass


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
    for _ in range(n):
        y = y_sampler()
        # FIXME: Seems inefficient and silly
        # Check if the observation space is single-valued, convert to list if so to make unpacking not throw an error
        try:
            iter(y)
        except TypeError:
            y = [y]
        # Initialize accumulators for information gain and marginal probability and associated normalization constants
        ig, ig_norm, marg, marg_norm = 0, 0, 0, 0
        # Across 'm' latent variable space samples
        for _ in range(m):
            i = theta_sampler()
            # FIXME: Seems inefficient and silly
            # Check if the latent space is single-valued, convert to list if so to make unpacking not throw an error
            try:
                iter(i)
            except TypeError:
                i = [i]
            # Compute the probabilities for the latent variable space sample [p(theta)] and observation given the latent
            # variable space sample [p(y|theta)], noting that this is implicitly p(y|theta, d) where d is the experiment
            p_i = np.exp(p_prior(*i))
            p_cond = np.exp(p_likely(*i, y))
            # Contribute to the estimate for the marginal probability p(y|d)
            marg += p_cond * p_i
            marg_norm += p_i
            # As long as the likelihood is non-zero, contribute to the estimate for the information gain IG(y, d)
            if p_cond > 0:
                ig += np.log(p_cond) * p_i
                ig_norm += p_i

        # Normalize the estimates for the information gain IG(y, d) and the marginal probability p(y|d)
        marg = marg / marg_norm if marg_norm > 0 else 0
        if marg > 0:
            ig -= np.log(marg) * ig_norm
        ig = ig / ig_norm if ig_norm > 0 else 0
        # With completed estimates for IG(y, d) and p(y|d) we can contribute to the estimate for EIG(d)
        eig += ig * marg
        eig_norm += marg

    # Normalize the estimate for EIG(d) and return it
    return eig / eig_norm if eig_norm > 0 else 0.0


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
        exp_handle.set_value(d)
        # Now we can estimate EIG(d) and add it to the list
        eig = eig_sampled(n, m, ltnt_sampler, obs_sampler, logp_prior, logp_likely)
        eig_pairs.append([d, eig])
        # Track which is the best experiment from an expected information gain perspective
        if i_max is None or eig > eig_pairs[i_max][1]:
            i_max = i

    # Return either just the best design found or a full dataframe reporting the designs and EIGs
    return eig_pairs[i_max] if return_best_only else pd.DataFrame(eig_pairs, columns=['design', 'eig'])
