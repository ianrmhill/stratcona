# Copyright (c) 2024 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import jax.numpy as jnp
import jax.random as rand
from jax.scipy.special import logsumexp
import scipy

from progress.bar import Bar

from numpyro.infer import NUTS, MCMC
import numpyro.distributions as dists
from numpyro.diagnostics import effective_sample_size, split_gelman_rubin

from stratcona.assistants.dist_translate import npyro_to_scipy
from stratcona.engine.bed import est_lp_y_g_x


def custom_mhgibbs_resampled_v(rng_key, spm, d, y, num_chains, n_v, beta=0.5):
    n_x = num_chains
    chain_len, warmup_len = 2200, 200
    k, kx, kv = rand.split(rng_key, 3)
    y_t = {}
    for exp in y:
        y_t |= {f'{exp}_{y_e}': jnp.expand_dims(y[exp][y_e], 0) for y_e in y[exp]}
    x_s = spm.sample(kx, d, num_samples=(n_x,), keep_sites=spm.hyls)
    lp_x = spm.logp(kx, d, site_vals=x_s, conditional=None, dims=(n_x,))
    # Compute the posterior probability of each sample in x_s
    lp_y_g_x, stats = est_lp_y_g_x(kv, spm, d, x_s, y_t, n_v)
    lp_x_likely = lp_x + lp_y_g_x.flatten()

    accept_percent = 0.0
    x_s_store = {hyl: jnp.array((), jnp.float32) for hyl in spm.hyls}
    bar = Bar('Sampling', max=chain_len)
    for i in range(chain_len):
        k, k1, k2, k3, k4 = rand.split(k, 5)
        # Determine a new set of samples xp_s by random walk of one hyper-latent
        # Uniformly choose one of the hyper-latent variables
        i_hyl = rand.choice(k1, jnp.arange(len(spm.hyls)))
        # Take a random step in some direction
        walks = dists.Normal(0.0, beta).sample(k2, x_s[spm.hyls[i_hyl]].shape)
        new_s = spm.hyl_info[spm.hyls[i_hyl]]['transform_inv'](x_s[spm.hyls[i_hyl]]) + walks
        xp_s_hyl = spm.hyl_info[spm.hyls[i_hyl]]['transform'](new_s)
        xp_s = x_s.copy()
        xp_s[spm.hyls[i_hyl]] = xp_s_hyl

        # Evaluate the posterior probability of the new samples
        lp_y_g_xp, stats_p = est_lp_y_g_x(k3, spm, d, xp_s, y_t, n_v)
        lp_xp_likely = lp_x + lp_y_g_xp.flatten()

        # Accept or reject the new samples
        p_accept = jnp.exp(lp_xp_likely - lp_x_likely)
        a_s = dists.Uniform(0, 1).sample(k4, (n_x,))
        accepted = jnp.where(jnp.greater(p_accept, a_s), True, False)
        lp_x_likely = jnp.where(accepted, lp_xp_likely, lp_x_likely)
        for hyl in spm.hyls:
            x_s[hyl] = x_s[hyl] if spm.hyls[i_hyl] != hyl else jnp.where(accepted, xp_s[hyl], x_s[hyl])
            # Add to final set of samples if warmup complete
            if i >= warmup_len:
                x_s_store[hyl] = jnp.append(x_s_store[hyl], x_s[hyl])
        # Ongoing performance statistics
        accept_percent = ((accept_percent * i) + (jnp.count_nonzero(accepted) / n_x)) / (i + 1)
        bar.next()
    bar.finish()

    # Fit the posterior distributions
    as_np = np.array(x_s_store['a0_nom'])
    new_prior = {}
    for hyl in spm.hyl_info:
        new_prior[hyl] = fit_dist_to_samples(spm.hyl_info[hyl], x_s_store[hyl])
    perf_stats = {'accept_percent': accept_percent}
    return new_prior, perf_stats


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
    data = hyl_info['transform_inv'](samples.flatten())
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
