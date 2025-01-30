# Copyright (c) 2025 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import jax.random as rand
from jax.scipy.special import logsumexp
import numpyro.distributions as dists
import numpy as np

import stratcona


def test_resampling_v():
    # Generate some observed data
    u, v1, v2, v3 = 2.3, 1.6, 0.5, 0.7
    d = dists.Normal(u, v1)
    d_v2 = dists.Normal(0, v2)
    d_v3 = dists.Normal(0, v3)
    k = rand.key(557393)
    k, k1, k2, k3, k4, k5, k6 = rand.split(k, 7)
    n_dev, n_chp, n_lot = 100, 1, 1
    #measd = d.sample(k1, (1, n_dev, n_chp, n_lot)) + d_v2.sample(k2, (1, 1, n_chp, n_lot)) + d_v3.sample(k3, (1, 1, 1, n_lot))
    measd = d.sample(k1, (1, n_dev, n_chp, n_lot))
    print(f'y - mean: {jnp.mean(measd)}, dev: {jnp.std(measd)}')

    var_tf = dists.transforms.ComposeTransform([dists.transforms.SoftplusTransform(), dists.transforms.AffineTransform(0, 0.1)])
    # Define a simple probabilistic model
    mb = stratcona.SPMBuilder(mdl_name='hierarchical_test')
    #mb.add_hyperlatent('u', dists.Normal, {'loc': 2.2, 'scale': 0.2})
    mb.add_hyperlatent('v1', dists.Normal, {'loc': 12, 'scale': 2}, transform=var_tf)
    #mb.add_hyperlatent('v2', dists.Normal, {'loc': 7, 'scale': 2}, transform=var_tf)
    #mb.add_hyperlatent('v3', dists.Normal, {'loc': 6, 'scale': 2}, transform=var_tf)
    #mb.add_latent('y', nom='u', dev='v1', chp='v2', lot='v3')
    mb.add_latent('y', nom='u', dev='v1')
    mb.add_params(obs_var=0.2, u=2.2)
    mb.add_observed('y_obs', dists.Normal, {'loc': 'y', 'scale': 'obs_var'}, n_dev)
    am = stratcona.AnalysisManager(mb.build_model(), rng_seed=253045)
    tst = stratcona.ReliabilityTest({'t1': {'lot': n_lot, 'chp': n_chp}}, {'t1': {}})
    am.set_test_definition(tst)

    # Sampling of x values
    n_x, n_v, n_y = 25, 1_000, 1
    y_tiled = jnp.repeat(jnp.repeat(jnp.expand_dims(measd, axis=(0, 1)), n_v, axis=1), n_x, axis=0)
    x_s = am.relmdl.sample(k4, am.test, (n_x,), keep_sites=am.relmdl.hyls)
    x_s_tiled = {x: jnp.repeat(jnp.repeat(jnp.expand_dims(x_s[x], (1, 2)), n_v, axis=1), n_y, axis=2) for x in x_s}

    ##################################
    # V resampling procedure
    ##################################
    # TODO: Can I share the n_v samples across n_y samples, resampling done individually for each y?
    v_init = am.relmdl.sample(k5, am.test, num_samples=(n_x, n_v, n_y), keep_sites=am.relmdl._ltnt_subsamples, conditionals=x_s_tiled)
    v_dev_zeros = {v: jnp.zeros_like(v_init[v]) for v in v_init if '_dev' in v}
    v_chp_zeros = {v: jnp.zeros_like(v_init[v]) for v in v_init if '_chp' in v}
    lot_ltnts = [ltnt for ltnt in v_init if '_lot' in ltnt]

    # First perform lot-level resampling
    v_s_lot_init = {ltnt: v_init[ltnt] for ltnt in lot_ltnts}
    # Get the log probabilities without summing so each lot can be considered individually
    lp_y_g_xv = am.relmdl.logp(k1, am.test, site_vals={'t1_y_obs': y_tiled}, conditional=x_s_tiled | v_s_lot_init | v_dev_zeros | v_chp_zeros, dims=(n_x, n_v, n_y), sum_lps=False)
    # Number of devices might be different for each observed variable, so have to sum across devices and chips
    lp_y_g_xv_lot = {y: jnp.sum(lp_y_g_xv[y], axis=(3, 4)) for y in lp_y_g_xv}
    # Element-wise addition of log-probabilities across different observe variables now that the dimensions match
    lp_y_g_xv_lot_tot = sum(lp_y_g_xv_lot.values())

    # Sum of all p_marg array elements must be 1 for resampling via random choice
    resample_probs = lp_y_g_xv_lot_tot - logsumexp(lp_y_g_xv_lot_tot, axis=1, keepdims=True)
    # Resample according to relative likelihood, need to resample indices so that resamples are the same for each lot-level latent variable
    inds_array = jnp.repeat(jnp.repeat(jnp.repeat(jnp.expand_dims(jnp.arange(n_v), (0, 2, 3)), n_x, axis=0), n_y, axis=2), n_lot, axis=3)
    # Spectacular triple vmap for n_x, n_y, and n_lot dimensions
    choice_vect = jax.vmap(jax.vmap(jax.vmap(rand.choice, (0, 0, None, None, 0), 0), (1, 2, None, None, 2), 2), (2, 3, None, None, 3), 3)
    k, kc = rand.split(k)
    krs = jnp.reshape(rand.split(kc, n_x * n_y * n_lot), (n_x, n_y, n_lot))
    resample_inds = choice_vect(krs, inds_array, (n_v,), True, jnp.exp(resample_probs))

    def reindex(a, i):
        return a[i]
    vec_reindex = jax.vmap(jax.vmap(jax.vmap(reindex, (0, 0), 0), (2, 2), 2), (3, 3), 3)
    v_rs_lot = {v: vec_reindex(v_s_lot_init[v], resample_inds) for v in lot_ltnts}
    #rs_diversity_lot = jnp.unique(v_rs_lot['t1_y_lot']).size / (n_x * n_v * n_y * n_lot)

    # Next up are chip-level variables
    chp_ltnts = [ltnt for ltnt in v_init if '_chp' in ltnt]
    v_s_chp_init = {ltnt: v_init[ltnt] for ltnt in chp_ltnts}
    # Get the log probabilities without summing so each lot can be considered individually
    lp_y_g_xv = am.relmdl.logp(k1, am.test, site_vals={'t1_y_obs': y_tiled}, conditional=x_s_tiled | v_rs_lot | v_dev_zeros | v_s_chp_init, dims=(n_x, n_v, n_y), sum_lps=False)
    # Number of devices might be different for each observed variable, so have to sum across devices
    lp_y_g_xv_chp = {y: jnp.sum(lp_y_g_xv[y], axis=3) for y in lp_y_g_xv}
    # Element-wise addition of log-probabilities across different observe variables now that the dimensions match
    lp_y_g_xv_chp_tot = sum(lp_y_g_xv_chp.values())

    # Sum of all p_marg array elements must be 1 for resampling via random choice
    resample_probs = lp_y_g_xv_chp_tot - logsumexp(lp_y_g_xv_chp_tot, axis=1, keepdims=True)
    # Resample according to relative likelihood, need to resample indices so that resamples are the same for each chip-level latent variable
    inds_array = jnp.repeat(jnp.repeat(jnp.repeat(jnp.repeat(jnp.expand_dims(jnp.arange(n_v), (0, 2, 3, 4)), n_x, axis=0), n_y, axis=2), n_chp, axis=3), n_lot, axis=4)
    # Spectacular quadruple vmap for n_x, n_y, n_chp, and n_lot dimensions
    choice_vect = jax.vmap(jax.vmap(jax.vmap(jax.vmap(rand.choice, (0, 0, None, None, 0), 0), (1, 2, None, None, 2), 2), (2, 3, None, None, 3), 3), (3, 4, None, None, 4), 4)
    k, kc = rand.split(k)
    krs = jnp.reshape(rand.split(kc, n_x * n_y * n_chp * n_lot), (n_x, n_y, n_chp, n_lot))
    resample_inds = choice_vect(krs, inds_array, (n_v,), True, jnp.exp(resample_probs))

    vec_reindex = jax.vmap(jax.vmap(jax.vmap(jax.vmap(reindex, (0, 0), 0), (2, 2), 2), (3, 3), 3), (4, 4), 4)
    v_rs_chp = {v: vec_reindex(v_s_chp_init[v], resample_inds) for v in chp_ltnts}
    #rs_diversity_chp = jnp.unique(v_rs_chp['t1_y_chp']).size / (n_x * n_v * n_y * n_chp * n_lot)

    # Finally are device-level variables
    dev_ltnts = [ltnt for ltnt in v_init if '_dev' in ltnt]
    v_s_dev_init = {ltnt: v_init[ltnt] for ltnt in dev_ltnts}
    # Get the log probabilities without summing so each lot can be considered individually
    lp_y_g_xv = am.relmdl.logp(k1, am.test, site_vals={'t1_y_obs': y_tiled}, conditional=x_s_tiled | v_rs_lot | v_s_dev_init | v_rs_chp, dims=(n_x, n_v, n_y), sum_lps=False)

    resample_inds = {}
    v_rs_dev = {}
    n_dev_tot = 0
    # Spectacular quintuple vmap for n_x, n_y, n_dev, n_chp, and n_lot dimensions
    choice_vect = jax.vmap(jax.vmap(jax.vmap(jax.vmap(jax.vmap(rand.choice, (0, 0, None, None, 0), 0), (1, 2, None, None, 2), 2), (2, 3, None, None, 3), 3), (3, 4, None, None, 4), 4), (4, 5, None, None, 5), 5)
    vec_reindex = jax.vmap(jax.vmap(jax.vmap(jax.vmap(jax.vmap(reindex, (0, 0), 0), (2, 2), 2), (3, 3), 3), (4, 4), 4), (5, 5), 5)

    for y in lp_y_g_xv:
        n_dev = lp_y_g_xv[y].shape[3]
        n_dev_tot += n_dev
        # Sum of all p_marg array elements must be 1 for resampling via random choice
        resample_probs = lp_y_g_xv[y] - logsumexp(lp_y_g_xv[y], axis=1, keepdims=True)
        # Resample according to relative likelihood, need to resample indices so that resamples are the same for each chip-level latent variable
        inds_array = jnp.repeat(jnp.repeat(jnp.repeat(jnp.repeat(jnp.repeat(jnp.expand_dims(jnp.arange(n_v), (0, 2, 3, 4, 5)), n_x, axis=0), n_y, axis=2), n_dev, axis=3), n_chp, axis=4), n_lot, axis=5)

        k, kc = rand.split(k)
        krs = jnp.reshape(rand.split(kc, n_x * n_y * n_dev * n_chp * n_lot), (n_x, n_y, n_dev, n_chp, n_lot))
        resample_inds[y] = choice_vect(krs, inds_array, (n_v,), True, jnp.exp(resample_probs))
        v_rs_dev |= {v: vec_reindex(v_s_dev_init[v], resample_inds[y]) for v in dev_ltnts if y in v}

    rs_diversity_dev = jnp.unique(v_rs_dev['t1_y_obs_y_dev']).size / (n_x * n_v * n_y * n_dev_tot * n_chp * n_lot)

    # Final logp
    lp_y_g_xv = am.relmdl.logp(k1, am.test, site_vals={'t1_y_obs': y_tiled}, conditional=x_s_tiled | v_rs_lot | v_rs_chp | v_rs_dev, dims=(n_x, n_v, n_y))
    lp_v_g_x = am.relmdl.logp(k1, am.test, site_vals=v_rs_dev | v_rs_chp | v_rs_lot, conditional=x_s_tiled, dims=(n_x, n_v, n_y))
    np_lp_y_g_xv = np.array(lp_y_g_xv)
    np_lp_v_g_x = np.array(lp_v_g_x)
    norm = logsumexp(lp_v_g_x - lp_y_g_xv, axis=1)
    #norm = jnp.log(n_v)
    lp_y_g_x_alt = logsumexp(lp_y_g_xv + lp_v_g_x, axis=1) - logsumexp(lp_y_g_xv, axis=1)

    v_std = jnp.std(v_rs_dev['t1_y_obs_y_dev'], axis=3)

    # Compute how similar the log prob of each lp_y_g_xv is, want to reduce the spread so that each sample is comparable
    lp_uniformity = 0

    # We want to get to this as the end result, successful marginalization over 'v' aleatoric uncertainty variable space
    lp_y_g_x = logsumexp(lp_v_g_x, axis=1)
    lp_y_g_x = lp_y_g_x - norm
    max_ind = jnp.argmax(lp_y_g_x)
    print(f'Log prob: {lp_y_g_x[max_ind]}')
    for x in x_s:
        print(f'{x}: {x_s[x][max_ind]}')
    print(x_s)
    print(lp_y_g_x)


def test_rs_func():
    # Generate some observed data
    u, v1, v2, v3 = 2.3, 1.6, 0.5, 0.7
    d = dists.Normal(u, v1)
    d_v2 = dists.Normal(0, v2)
    d_v3 = dists.Normal(0, v3)
    k = rand.key(557393)
    k, k1, k2 = rand.split(k, 3)
    n_dev, n_chp, n_lot = 100, 1, 1
    #measd = d.sample(k1, (1, n_dev, n_chp, n_lot)) + d_v2.sample(k2, (1, 1, n_chp, n_lot)) + d_v3.sample(k3, (1, 1, 1, n_lot))
    measd = d.sample(k1, (1, n_dev, n_chp, n_lot))
    print(f'y - mean: {jnp.mean(measd)}, dev: {jnp.std(measd)}')

    var_tf = dists.transforms.ComposeTransform([dists.transforms.SoftplusTransform(), dists.transforms.AffineTransform(0, 0.1)])
    # Define a simple probabilistic model
    mb = stratcona.SPMBuilder(mdl_name='hierarchical_test')
    mb.add_hyperlatent('u', dists.Normal, {'loc': 2.2, 'scale': 0.2})
    mb.add_hyperlatent('v1', dists.Normal, {'loc': 12, 'scale': 2}, transform=var_tf)
    #mb.add_hyperlatent('v2', dists.Normal, {'loc': 7, 'scale': 2}, transform=var_tf)
    #mb.add_hyperlatent('v3', dists.Normal, {'loc': 6, 'scale': 2}, transform=var_tf)
    #mb.add_latent('y', nom='u', dev='v1', chp='v2', lot='v3')
    mb.add_latent('y', nom='u', dev='v1')
    mb.add_params(obs_var=0.2)
    mb.add_observed('y_obs', dists.Normal, {'loc': 'y', 'scale': 'obs_var'}, n_dev)
    am = stratcona.AnalysisManager(mb.build_model(), rng_seed=253045)
    tst = stratcona.ReliabilityTest({'t1': {'lot': n_lot, 'chp': n_chp}}, {'t1': {}})
    am.set_test_definition(tst)

    # Sampling of x values
    n_x, n_v, n_y = 25, 1_000, 1
    x_s = am.relmdl.sample(k2, am.test, (n_x,), keep_sites=am.relmdl.hyls)

    lp_y_g_x, stats = stratcona.engine.bed.est_lp_y_g_x(k, am.relmdl, am.test, x_s, {'t1_y_obs': measd}, n_v)
    print(lp_y_g_x)
