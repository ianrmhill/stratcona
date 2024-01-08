# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import pymc
import arviz
import numpy as np
import scipy

from matplotlib import pyplot as plt

from stratcona.assistants.dist_translate import pymc_to_scipy

__all__ = ['inference_model', 'fit_latent_params_to_posterior_samples']


def inference_model(model, num_samples: int = None, num_chains: int = None, seeding: tuple = None):
    # TODO: Would make sense to have this function or its parent take the observed data as an argument and set it using
    #       the shared variable handle

    # Set up a dictionary of all the optional arguments that we may want to override the defaults for
    extra_args = {}
    if seeding:
        # To force the random draws during inference we need to ensure a random seed is provided for each chain
        if num_chains and not len(seeding) == num_chains:
            raise ValueError(f"Number of seeds in tuple must match the number of MCMC chains.")
        elif not num_chains:
            num_chains = len(seeding)
        extra_args['random_seed'] = seeding
    if num_chains:
        extra_args['chains'] = num_chains
    if num_samples:
        extra_args['draws'] = num_samples

    # Now run the MCMC sampling to get a sample trace of the posterior
    trace = pymc.sample(model=model, **extra_args)
    # TODO: Interpret the MCMC convergence statistics to give the user recommendations to improve the model
    return trace


def fit_latent_params_to_posterior_samples(latents: dict, idata: arviz.InferenceData, run_fit_analysis=False):
    """
    After PyMC runs a sampling algorithm we end up with many samples from the posterior. What PyMC does not do, however,
    is update the parameterized latent variables based on the posterior. Here we do this manually using the generated
    samples.

    Returns
    -------

    """
    post_data = idata.posterior
    posterior_params = {}
    for ltnt in latents:
        sampled = post_data[ltnt].values.flatten()
        dist_to_fit = pymc_to_scipy(latents[ltnt].dist.rv_op.name)
        scipy_dist = getattr(scipy.stats, dist_to_fit)
        params = scipy_dist.fit(sampled)
        posterior_params[ltnt] = {'mu': params[0], 'sigma': params[1]}
        if run_fit_analysis:
            check_fit_quality(ltnt, sampled, scipy_dist, posterior_params[ltnt])

    return posterior_params


def check_fit_quality(variable_name, data, dist_type, dist_params):
    """
    Given a dataset and a fitted distribution, we can evaluate whether the fit is decent by using a relative check. If
    any other distribution gives a better fit, let the user know that a different distribution may be a better choice
    to represent the variable PDF.

    Parameters
    ----------

    Returns
    -------

    """
    kde = scipy.stats.gaussian_kde(data)
    # Now check the fits against each other
    x = np.linspace(np.min(data), np.max(data), 100)

    # Error types: sum of square errors, RSS/SSE, Wasserstein, Kolmogorov-Smirnov (KS), or Energy
    # One nice Bayesian way that aligns with objectives: quantiles matching estimation (QME)

    # Plotting sanity checks
    #f, p = plt.subplots(figsize=(8, 2))
    #p.hist(data, bins=100, density=True, color='grey')
    #p.plot(x, kde(x), color='blue')
    #params = dist_params.values()
    #p.plot(x, dist_type.pdf(x, *params), color='green')
    #plt.show()
