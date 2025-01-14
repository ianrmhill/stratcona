# Copyright (c) 2024 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import jax.numpy as jnp
import scipy

from numpyro.infer import SA, BarkerMH, NUTS, MCMC
from numpyro.diagnostics import effective_sample_size, split_gelman_rubin

from stratcona.assistants.dist_translate import npyro_to_scipy


def inference_model(model, hyl_info, observed_data, rng_key, num_samples: int = 10_000, num_chains: int = 4):
    kernel = NUTS(model)
    sampler = MCMC(kernel, num_warmup=2_000, num_samples=num_samples, num_chains=num_chains, progress_bar=True)
    sampler.run(rng_key, measured=observed_data, extra_fields=('potential_energy',))
    samples = sampler.get_samples(group_by_chain=True)

    convergence_stats = {}
    for site in samples:
        convergence_stats[site] = {'ess': effective_sample_size(samples[site]), 'srhat': split_gelman_rubin(samples[site])}
    extra_info = sampler.get_extra_fields()
    diverging = extra_info['diverging'] if 'diverging' in extra_info else 0
    diverging = jnp.sum(diverging)
    # TODO: Interpret the MCMC convergence statistics to give the user recommendations to improve the model
    print(convergence_stats)
    print(f'Divergences: {diverging}')

    new_prior = {}
    for hyl in hyl_info:
        new_prior[hyl] = fit_dist_to_samples(hyl_info[hyl], samples[hyl])
    return new_prior


def fit_dist_to_samples(hyl_info, samples):
    """Fits a numpyro distribution's parameters to a set of sampled values using MLE methods."""
    # Apply the inverse of any transforms of the hyl base distribution to the data, otherwise the base distribution
    # is erroneously fit to the transformed data instead
    data = hyl_info['scale'](samples.flatten())
    dist, prm_names, prm_transforms, fit_kwargs = npyro_to_scipy(hyl_info['dist'])
    prms = dist.fit(data, **fit_kwargs)
    npyro_prms = {}
    for i, val in enumerate(prms):
        if prm_names[i] is not None:
            npyro_prms[prm_names[i]] = prm_transforms[i](val)
    if hyl_info['fixed'] is not None:
        for prm in hyl_info['fixed']:
            npyro_prms[prm] = hyl_info['fixed'][prm]
    return npyro_prms


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
