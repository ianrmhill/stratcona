# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import pymc
import arviz
import numpy as np
import jax.numpy as jnp
import scipy

from numpyro.infer import NUTS, MCMC
from numpyro.diagnostics import effective_sample_size, split_gelman_rubin

from multiprocess import Pool
from functools import partial

from matplotlib import pyplot as plt

from stratcona.assistants.dist_translate import pymc_to_scipy, npyro_to_scipy
from stratcona.modelling.relmodel import ReliabilityModel

__all__ = ['inference_model', 'fit_latent_params_to_posterior_samples', 'importance_sampling_inference']


def importance_sampling_chain(y, m, i_s, lp_i, lp_y):
    test_i = i_s()
    i_store = np.empty((m, len(test_i), len(test_i[0])))
    lp_i_store = np.zeros((m,))
    lp_w_store = np.zeros((m,))

    ### Computation ###
    for m_i in range(m):
        i = i_s()
        i_store[m_i] = i
        lp_i_store[m_i] = lp_i(*i)
        lp_w_store[m_i] = lp_y(*i, *y) + lp_i_store[m_i]

    return i_store, lp_w_store


def importance_sampling_inference(y, latents, prm_map, i_s, lp_i, lp_y, num_samples: int = 3000, num_chains: int = 4,
                                  multicore: bool = False, resample_rng=None):
    ### Setup Work ###
    if multicore:
        pool = Pool(processes=4)
    # Provides the capability to test the routine through reproducible sampling
    rng = np.random.default_rng() if resample_rng is None else resample_rng

    to_run = partial(importance_sampling_chain, m=num_samples, y=list(y.values()), i_s=i_s, lp_i=lp_i, lp_y=lp_y)

    if multicore:
        with pool:
            outs = pool.map(to_run)
    else:
        outs = []
        for chain in range(num_chains):
            outs.append(to_run())

    i_list = [outs[i][0] for i in range(num_chains)]
    samples = np.concatenate(i_list)
    w_list = [outs[i][1] for i in range(num_chains)]
    weights = np.exp(np.concatenate(w_list))
    # Sum of all p_marg array elements must be 1 for resampling via numpy's choice
    weights_normed = weights / np.sum(weights)
    n = num_samples * num_chains
    # Compute the effective sample size as a diagnostic
    ess = 1 / np.sum(weights_normed ** 2)
    print(f"ESS = {ess}")
    # Resample according to the marginal likelihood of each sample
    resampled_inds = rng.choice(n, (n,), p=weights_normed)
    print(f"Resample diversity = {100 * (len(np.unique(resampled_inds)) / n)}, count = {len(np.unique(resampled_inds))}")
    test_i = i_s()
    resamples = np.empty((n, len(test_i), len(test_i[0])))
    for s_i in range(n):
        resamples[s_i] = samples[resampled_inds[s_i]]

    fit_samples = {}
    for i, ltnt in enumerate(latents):
        fit_samples[ltnt.name] = resamples[:, i]
    return fit_latent_params_to_posterior_samples(latents, prm_map, fit_samples)


def inference_model(model: ReliabilityModel, hyl_info, observed_data, rng_key, num_samples: int = 2_000, num_chains: int = 4):
    kernel = NUTS(model)
    sampler = MCMC(kernel, num_warmup=1_000, num_samples=num_samples, num_chains=num_chains)
    sampler.run(rng_key, measured=observed_data, extra_fields=('potential_energy',))
    samples = sampler.get_samples(group_by_chain=True)

    convergence_stats = {}
    for site in samples:
        convergence_stats[site] = {'ess': effective_sample_size(samples[site]), 'srhat': split_gelman_rubin(samples[site])}
    extra_info = sampler.get_extra_fields()
    diverging = extra_info['diverging'] if 'diverging' in extra_info else 0
    diverging = jnp.sum(diverging)
    # TODO: Interpret the MCMC convergence statistics to give the user recommendations to improve the model
    #print(convergence_stats)
    #print(f'Divergences: {diverging}')

    new_prior = {}
    for hyl in hyl_info:
        new_prior[hyl] = fit_dist_to_samples(hyl_info[hyl]['dist'], samples[hyl], fixed_prms=hyl_info[hyl]['fixed'])
    return new_prior


def fit_dist_to_samples(numpyro_dist, samples, fixed_prms=None):
    """Fits a numpyro distribution's parameters to a set of sampled values using MLE methods."""
    data = samples.flatten()
    dist, prm_names, prm_transforms, fit_kwargs = npyro_to_scipy(numpyro_dist)
    prms = dist.fit(data, **fit_kwargs)
    npyro_prms = {}
    for i, val in enumerate(prms):
        if prm_names[i] is not None:
            npyro_prms[prm_names[i]] = prm_transforms[i](val)
    if fixed_prms is not None:
        for prm in fixed_prms:
            npyro_prms[prm] = fixed_prms[prm]
    # FIXME: Need to know the transformations applied to back-calculate the posterior params as the scipy dists are not transformed
    return npyro_prms


def fit_latent_params_to_posterior_samples(latents: list, prm_map: dict, idata: arviz.InferenceData | dict[np.ndarray],
                                           run_fit_analysis=False):
    """
    After PyMC runs a sampling algorithm we end up with many samples from the posterior. What PyMC does not do, however,
    is update the parameterized latent variables based on the posterior. Here we do this manually using the generated
    samples.

    Returns
    -------

    """
    if type(idata) == arviz.InferenceData:
        post_data = idata.posterior
    else:
        post_data = idata
    posterior_params = {}
    for ltnt in latents:
        if type(idata) == arviz.InferenceData:
            sampled = post_data[ltnt.name].values.flatten()
        else:
            sampled = post_data[ltnt.name]
        dist_to_fit = pymc_to_scipy(ltnt.owner.op.name)
        if dist_to_fit == 'categorical':
            # For discrete variables we have to perform the updates manually. We transform everything into a
            # categorical distribution to form a distribution that can fit any sampled dataset perfectly.
            total_samples = sampled.size
            # Count the number of occurrences of each value then normalize to get the posterior probabilities
            updated = np.bincount(sampled) / total_samples
            posterior_params[ltnt.name] = {'p': updated}
        else:
            scipy_dist = getattr(scipy.stats, dist_to_fit)
            params = scipy_dist.fit(sampled)
            new_prms = {}
            if dist_to_fit == 'gamma':
                new_prms['alpha'] = params[0]
                new_prms['beta'] = 1 / params[2]
            else:
                for i, prm in enumerate(prm_map[ltnt.name]):
                    new_prms[prm] = params[i]
            posterior_params[ltnt.name] = new_prms
            # Fit analysis only useful for continuous variables since the categorical distribution will fit every
            # set of discrete samples perfectly
            if run_fit_analysis:
                check_fit_quality(ltnt.name, sampled, scipy_dist, posterior_params[ltnt])

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
