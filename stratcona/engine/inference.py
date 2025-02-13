# Copyright (c) 2025 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import jax.random as rand
from jax.scipy.special import logsumexp

from progress.bar import Bar
from functools import partial
from math import prod
import numpy as np

from numpyro.infer import NUTS, MCMC
import numpyro.distributions as dists
from numpyro.diagnostics import effective_sample_size, split_gelman_rubin

from stratcona.assistants.dist_translate import npyro_to_scipy
from stratcona.engine.bed import est_lp_y_g_x
from stratcona.modelling.relmodel import ReliabilityTest, TestDef, ExpDims


@partial(jax.jit, static_argnames=['spm', 'test_dims', 'batch_dims', 'g_hyl'])
def mhgibbs_inner_loop(rng_key, spm, test_dims, test_conds, x_s, y_t, beta, lp_x, lp_prev, batch_dims, g_hyl):
    k2, k3, k4 = rand.split(rng_key, 3)
    # Determine a new set of samples xp_s by random walk of one hyper-latent
    # Take a random step in some direction
    walks = dists.Normal(0.0, beta).sample(k2, x_s[g_hyl].shape)
    new_s = spm.hyl_info[g_hyl]['transform_inv'](x_s[g_hyl]) + walks
    xp_s_hyl = spm.hyl_info[g_hyl]['transform'](new_s)
    xp_s = x_s.copy()
    xp_s[g_hyl] = xp_s_hyl

    # Evaluate the posterior probability of the new samples
    lp_y_g_xp, stats_p = int_out_v(k3, spm, batch_dims, test_dims, test_conds, xp_s, y_t)
    lp_xp_likely = lp_x + lp_y_g_xp.flatten()

    # Accept or reject the new samples
    p_accept = jnp.exp(lp_xp_likely - lp_prev)
    a_s = dists.Uniform(0, 1).sample(k4, (batch_dims[0],))
    accepted = jnp.where(jnp.greater(p_accept, a_s), True, False)
    lp_new = jnp.where(accepted, lp_xp_likely, lp_prev)
    x_new = x_s.copy()
    x_new[g_hyl] = jnp.where(accepted, xp_s[g_hyl], x_s[g_hyl])
    # Ongoing performance statistics
    ap = jnp.count_nonzero(accepted) / batch_dims[0]
    return x_new, lp_new, ap


def custom_mhgibbs_new(rng_key, spm, d, y, num_chains, n_v, beta=0.5):
    n_x = num_chains
    chain_len, warmup_len = 2200, 200
    k, kx, kv = rand.split(rng_key, 3)
    y_t = {}
    for exp in y:
        y_t |= {f'{exp}_{y_e}': jnp.expand_dims(y[exp][y_e], 0) for y_e in y[exp]}
    x_s = spm.sample_new(kx, d.dims, d.conds, (n_x,), spm.hyls)
    lp_x = spm.logp_new(kx, d.dims, d.conds, x_s, None, (n_x,))
    # Compute the posterior probability of each sample in x_s
    batch_dims = (n_x, n_v, 1)
    lp_y_g_x, stats = int_out_v(kv, spm, batch_dims, d.dims, d.conds, x_s, y_t)
    lp_lkly = lp_x + lp_y_g_x.flatten()

    accept_percent = 0.0
    x_s_store = {hyl: jnp.array((), jnp.float32) for hyl in spm.hyls}
    bar = Bar('Sampling', max=chain_len)
    for i in range(chain_len):
        k, kc, ki = rand.split(k, 3)
        # Uniformly choose one of the hyper-latent variables
        g_hyl = spm.hyls[rand.choice(kc, jnp.arange(len(spm.hyls)))]
        x_s, lp_lkly, ap = mhgibbs_inner_loop(ki, spm, d.dims, d.conds, x_s, y_t, beta, lp_x, lp_lkly, batch_dims, g_hyl)
        # Add to final set of samples if warmup complete
        if i >= warmup_len:
            for hyl in spm.hyls:
                x_s_store[hyl] = jnp.append(x_s_store[hyl], x_s[hyl])
        accept_percent = ((accept_percent * i) + ap) / (i + 1)
        bar.next()

    bar.finish()
    # Fit the posterior distributions
    new_prior = {}
    for hyl in spm.hyl_info:
        new_prior[hyl] = fit_dist_to_samples(spm.hyl_info[hyl], x_s_store[hyl])
    perf_stats = {'accept_percent': accept_percent}
    return new_prior, perf_stats


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


def batch_choice(n_extra_dims):
    inner = jax.vmap(jax.vmap(rand.choice, (0, 0, None, None, 0), 0), (1, 2, None, None, 2), 2)
    for i in range(n_extra_dims):
        inner = jax.vmap(inner, (i + 1, i + 2, None, None, i + 2), i + 2)
    return inner


def batch_reindex(n_extra_dims):
    def reindex(a, i):
        return a[i]
    inner = jax.vmap(jax.vmap(reindex, (0, 0), 0), (2, 2), 2)
    for i in range(n_extra_dims):
        inner = jax.vmap(inner, (i + 2, i + 2), i + 2)
    return inner


@partial(jax.jit, static_argnames=['n', 'dims'])
def get_inds_arr(n, dims):
    add_dims = tuple([0, 2] + [i + 3 for i in range(len(dims) - 2)])
    inner = jnp.repeat(jnp.repeat(jnp.expand_dims(jnp.arange(n), add_dims), dims[0], axis=0), dims[1], axis=2)
    for i, d in enumerate(dims[2:]):
        inner = jnp.repeat(inner, d, axis=i+3)
    return inner


@jax.jit
def count_unique(x):
    """Credit to jakevdp for this approach: https://stackoverflow.com/questions/77082029/jax-count-unique-elements-in-array"""
    x = jnp.sort(x.flatten())
    return 1 + (x[1:] != x[:-1]).sum()


#@partial(jax.jit, static_argnames=['spm', 'test_dims', 'batch_dims'])
def int_out_v(rng_key, spm, batch_dims: (int, int, int), test_dims: frozenset[ExpDims], test_conds, x_s, y_s):
    n_x, n_v, n_y = batch_dims
    k, k_init, k_noise, kdum = rand.split(rng_key, 4)
    perf_stats = {}
    # Adjust the variance of y_s_t to compensate for measurement noise
    meas_noise_std = 0.04
    vscale = {y: jnp.sqrt(jnp.abs(jnp.std(y_s[y])**2 - meas_noise_std**2)) / jnp.std(y_s[y]) for y in y_s}
    y_s_a = {y: ((y_s[y] - jnp.mean(y_s[y])) * vscale[y]) + jnp.mean(y_s[y]) for y in y_s}
    print(f'Yadj std dev: {jnp.std(y_s_a["e_y"])}')
    # Tile the x and y sample arrays to match the problem dimensions for full vectorization
    x_s_t = {x: jnp.repeat(jnp.repeat(jnp.expand_dims(x_s[x], (1, 2)), n_v, axis=1), n_y, axis=2) for x in x_s}
    y_s_t = {y: jnp.repeat(jnp.repeat(jnp.expand_dims(y_s_a[y], axis=(0, 1)), n_v, axis=1), n_x, axis=0) for y in y_s_a}


    v_init = spm.sample_new(k_init, test_dims, test_conds, batch_dims, keep_sites=spm.ltnt_subsamples, conditionals=x_s_t)
    lp_v_init = spm.logp_new(kdum, test_dims, test_conds, v_init, x_s_t, batch_dims, sum_lps=False)

    v_dev_zeros = {v: jnp.zeros_like(v_init[v]) for v in v_init if '_dev' in v}
    v_chp_zeros = {v: jnp.zeros_like(v_init[v]) for v in v_init if '_chp' in v}
    v_rs = {'lot': {}, 'chp': v_chp_zeros, 'dev': v_dev_zeros}

    in_loops = {'lot': ['na'], 'chp': ['na'], 'dev': [y for y in y_s_t]}
    for i, lvl in enumerate(v_rs.keys()):
        lvl_ltnts = [ltnt for ltnt in v_init if f'_{lvl}' in ltnt]
        if len(lvl_ltnts) > 0:
            choice, reindex = batch_choice(i + 1), batch_reindex(i + 1)
            # Handle each experiment within the test separately
            for e in test_dims:
                e_ltnts = [ltnt for ltnt in lvl_ltnts if f'{e.name}_' in ltnt]
                # Get a subset of the test defined by test_dims and test_conds, the single experiment
                e_tst = TestDef(e.name, {e}, {e.name: test_conds[e.name]})
                e_y_s_t = {y: y_s_t[y] for y in y_s_t if f'{e.name}_' in y}
                v_s_init = {ltnt: v_init[ltnt] for ltnt in e_ltnts}
                v_rs[lvl] = v_s_init
                conditional = x_s_t | v_rs['lot'] | v_rs['chp'] | v_rs['dev']
                # Get the log probabilities without summing so each lot can be considered individually
                lp_y_g_xv = spm.logp_new(kdum, e_tst.dims, e_tst.conds, e_y_s_t, conditional, batch_dims, sum_lps=False)

                for yp in in_loops[lvl]:
                    # Number of devices might be different for each observed variable, so have to sum across devices and chips
                    if lvl == 'dev':
                        lp_v = sum([lp_v_init[ltnt] for ltnt in e_ltnts if yp in ltnt])
                        # Craziest step - resampling probabilities are adjusted based on the dimensionality (# of RVs) of v
                        # This is done to massively reduce the variance of the lp_y_g_x estimate when the dimensionality is high, as
                        # all sampled v need to give similar likelihood for y to avoid lucky samples dominating the estimates.
                        # Instead, all samples become very lucky on average, thus the variance decreases
                        # It does reduce the resample diversity, but less than one would intuitively expect, and can be compensated for
                        # by increasing n_v
                        # FIXME: Finding v samples so close to y causes measurement noise to leak into inference estimates of variability
                        lp_y_g_xv_tot = (lp_y_g_xv[yp] * (1 + jnp.log(100))) - lp_v #TODO: Auto determine the v dimension size
                        #lp_y_g_xv_tot = (lp_y_g_xv[yp]) * (1 + jnp.log(50)) + lp_v
                        #lp_y_g_xv_tot = lp_y_g_xv[yp] - lp_v
                        #lp_y_g_xv_tot = (lp_y_g_xv[yp]) * (1 + jnp.log(100))
                        #lp_y_g_xv_tot = lp_y_g_xv[yp]
                    else:
                        lp_v = sum([lp_v_init[ltnt] for ltnt in e_ltnts])
                        sum_axes = (3, 4) if lvl == 'lot' else 3
                        lp_y_g_xv = {y: jnp.sum(lp_y_g_xv[y], axis=sum_axes) for y in lp_y_g_xv}
                        # Element-wise addition of log-probabilities across different observe variables now that the dimensions match
                        # Subtract the sample probability p(v|x) for each to avoid biasing resamples towards the prior values
                        lp_y_g_xv_tot = (sum(lp_y_g_xv.values()) * (1 + jnp.log(25))) - lp_v # TODO: Auto determine the v dimension size


                    # Sum of all p_marg array elements must be 1 for resampling via random choice
                    resample_probs = lp_y_g_xv_tot - logsumexp(lp_y_g_xv_tot, axis=1, keepdims=True)
                    # Resample according to relative likelihood, need to resample indices so that resamples are the same for each lot-level latent variable
                    if lvl == 'lot':
                        dims = (n_x, n_y, e.lot)
                    elif lvl == 'chp':
                        dims = (n_x, n_y, e.chp, e.lot)
                    else:
                        dims = (n_x, n_y, lp_y_g_xv[yp].shape[3], e.chp, e.lot)
                    inds = get_inds_arr(n_v, dims)
                    k, ks = rand.split(k)
                    krs = jnp.reshape(rand.split(ks, prod(dims)), dims)
                    resample_inds = choice(krs, inds, (n_v,), True, jnp.exp(resample_probs))
                    v_rs[lvl] |= {v: reindex(v_s_init[v], resample_inds) for v in e_ltnts}

                if lvl == 'lot':
                    v_diversity = [count_unique(v_rs[lvl][v]) / (n_x * n_v * n_y * e.lot) for v in e_ltnts]
                elif lvl == 'chp':
                    v_diversity = [count_unique(v_rs[lvl][v]) / (n_x * n_v * n_y * e.chp * e.lot) for v in e_ltnts]
                else:
                    v_diversity = [count_unique(v_rs[lvl][v]) / (n_x * n_v * n_y * v_rs[lvl][v].shape[3] * e.chp * e.lot) for v in e_ltnts]
                perf_stats[f'{e.name}_{lvl}_rs_diversity'] = sum(v_diversity) / len(e_ltnts)

    # Final logp
    lp_v_g_x = spm.logp_new(kdum, test_dims, test_conds, v_rs['lot'] | v_rs['chp'] | v_rs['dev'], x_s_t, batch_dims)
    lp_v_np = np.array(lp_v_g_x)
    print(f'LP-V mean - {jnp.mean(lp_v_g_x)}, std dev - {jnp.std(lp_v_g_x)}, min - {jnp.min(lp_v_g_x)}, max - {jnp.max(lp_v_g_x)}')
    lp_y_g_xv = spm.logp_new(kdum, test_dims, test_conds, y_s_t, x_s_t | v_rs['lot'] | v_rs['chp'] | v_rs['dev'], batch_dims)
    lp_y_np = np.array(lp_y_g_xv)
    print(f'LP-Y mean - {jnp.mean(lp_y_g_xv)}, std dev - {jnp.std(lp_y_g_xv)}, min - {jnp.min(lp_y_g_xv)}, max - {jnp.max(lp_y_g_xv)}')
    lp_y_g_x = logsumexp(lp_v_g_x + lp_y_g_xv, axis=1)
    return lp_y_g_x, perf_stats


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
    kde = jax.scipy.stats.gaussian_kde(data)
    # Now check the fits against each other
    x = jnp.linspace(jnp.min(data), jnp.max(data), 100)

    # Error types: sum of square errors, RSS/SSE, Wasserstein, Kolmogorov-Smirnov (KS), or Energy
    # One nice Bayesian way that aligns with objectives: quantiles matching estimation (QME)

    # Plotting sanity checks
    #f, p = plt.subplots(figsize=(8, 2))
    #p.hist(data, bins=100, density=True, color='grey')
    #p.plot(x, kde(x), color='blue')
    #params = dist_params.values()
    #p.plot(x, dist_type.pdf(x, *params), color='green')
    #plt.show()
