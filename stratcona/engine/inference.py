# Copyright (c) 2024 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import jax.numpy as jnp
import jax.random as rand
from jax.scipy.special import logsumexp
import scipy

from numpyro.infer import SA, BarkerMH, NUTS, MCMC
from numpyro.diagnostics import effective_sample_size, split_gelman_rubin

from stratcona.assistants.dist_translate import npyro_to_scipy
from stratcona.engine.bed import est_lp_y_g_x


def custom_inference(rng_key, spm, d, y, n_x, n_v):
    k, kx, kv, kc = rand.split(rng_key, 4)
    y_t = {}
    for exp in y:
        y_t |= {f'{exp}_{y_e}': jnp.expand_dims(y[exp][y_e], 0) for y_e in y[exp]}
    x_s = spm.sample(kx, d, num_samples=(n_x,), keep_sites=spm.hyls)
    lp_x = spm.logp(kx, d, site_vals=x_s, conditional=None, dims=(n_x,))

    lp_y_g_x, stats = est_lp_y_g_x(kv, spm, d, x_s, y_t, n_v)
    lp_y = logsumexp(lp_y_g_x.flatten(), axis=0) - jnp.log(n_x)

    lp_x_g_y = (lp_x + lp_y_g_x.flatten()) - lp_y
    xgy_norm = logsumexp(lp_x_g_y)
    p_x_g_y = jnp.exp(lp_x_g_y - xgy_norm)
    # Resample to get a distribution of samples according to p(x|y,d)
    resample_inds = rand.choice(kc, jnp.arange(n_x), (n_x,), True, p_x_g_y)
    # Fit the posterior distributions
    new_prior = {}
    for hyl in spm.hyl_info:
        resamples = x_s[hyl][resample_inds]
        new_prior[hyl] = fit_dist_to_samples(spm.hyl_info[hyl], resamples)
    return new_prior


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
    #print(convergence_stats)
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
