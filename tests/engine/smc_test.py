# Copyright (c) 2025 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import jax.random as rand
from jax.scipy.special import logsumexp
import numpyro.distributions as dists

import stratcona


def test_resampling_v():
    # Generate some observed data
    u, v1, v2, v3 = 2.3, 1.0, 0.5, 0.7
    d = dists.Normal(u, v1)
    d_v2 = dists.Normal(0, v2)
    d_v3 = dists.Normal(0, v3)
    k = rand.key(557393)
    k1, k2, k3, k4, k5, k6 = rand.split(k, 6)
    n_dev, n_chp, n_lot = 4, 3, 2
    measd = d.sample(k1, (1, n_dev, n_chp, n_lot)) + d_v2.sample(k2, (1, 1, n_chp, n_lot)) + d_v3.sample(k3, (1, 1, 1, n_lot))

    # Define a simple probabilistic model
    mb = stratcona.SPMBuilder(mdl_name='hierarchical_test')
    mb.add_hyperlatent('u', dists.Normal, {'loc': 2.2, 'scale': 0.2})
    mb.add_hyperlatent('v1', dists.Normal, {'loc': 1.2, 'scale': 0.2}, transform=dists.transforms.SoftplusTransform())
    mb.add_hyperlatent('v2', dists.Normal, {'loc': 0.7, 'scale': 0.2}, transform=dists.transforms.SoftplusTransform())
    mb.add_hyperlatent('v3', dists.Normal, {'loc': 0.6, 'scale': 0.2}, transform=dists.transforms.SoftplusTransform())
    mb.add_latent('y', nom='u', dev='v1', chp='v2', lot='v3')
    mb.add_params(obs_var=0.2)
    mb.add_observed('y_obs', dists.Normal, {'loc': 'y', 'scale': 'obs_var'}, n_dev)
    am = stratcona.AnalysisManager(mb.build_model(), rng_seed=253045)
    tst = stratcona.ReliabilityTest({'t1': {'lot': n_lot, 'chp': n_chp}}, {'t1': {}})
    am.set_test_definition(tst)

    # Sampling of x values
    n_x, n_v, n_y = 3, 5, 1
    y_tiled = jnp.repeat(jnp.repeat(jnp.expand_dims(measd, axis=(0, 1)), n_v, axis=1), n_x, axis=0)
    x_s = am.relmdl.sample(k4, am.test, (n_x,), keep_sites=['u', 'v1', 'v2', 'v3'])
    x_s_tiled = {x: jnp.repeat(jnp.repeat(jnp.expand_dims(x_s[x], (1, 2)), n_v, axis=1), n_y, axis=2) for x in x_s}

    ##################################
    # V SMC Procedure
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
    krs = jnp.reshape(rand.split(k6, n_x * n_y * n_lot), (n_x, n_y, n_lot))
    resample_inds = choice_vect(krs, inds_array, (n_v,), True, jnp.exp(resample_probs))

    def reindex(a, i):
        return a[i]
    vec_reindex = jax.vmap(jax.vmap(jax.vmap(reindex, (0, 0), 0), (2, 2), 2), (3, 3), 3)
    v_rs_lot = {v: vec_reindex(v_s_lot_init[v], resample_inds) for v in lot_ltnts}
    resample_diversity_percent = jnp.unique(v_rs_lot['t1_y_lot']).size / (n_x * n_v * n_y * n_lot)

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
    krs = jnp.reshape(rand.split(k6, n_x * n_y * n_chp * n_lot), (n_x, n_y, n_chp, n_lot))
    resample_inds = choice_vect(krs, inds_array, (n_v,), True, jnp.exp(resample_probs))

    vec_reindex = jax.vmap(jax.vmap(jax.vmap(jax.vmap(reindex, (0, 0), 0), (2, 2), 2), (3, 3), 3), (4, 4), 4)
    v_rs_chp = {v: vec_reindex(v_s_chp_init[v], resample_inds) for v in chp_ltnts}
    resample_diversity_percent = jnp.unique(v_rs_chp['t1_y_chp']).size / (n_x * n_v * n_y * n_chp * n_lot)

    # Finally are device-level variables



    v_s_lot_init = am.relmdl.sample(k5, am.test, num_samples=(n_x, n_v, n_y), keep_sites=lot_ltnts, conditionals=x_s_tiled | v_dev_zeros | v_chp_zeros)

    # Final logp
    lp_y_g_xv = am.relmdl.logp(k1, am.test, site_vals={'t1_y_obs': y_tiled}, conditional=x_s_tiled_2 | v_s_tiled, dims=(n_x, n_v, n_y))
    # Compute how similar the log prob of each lp_y_g_xv is, want to reduce the spread so that each sample is comparable
    lp_uniformity = 0

    # We want to get to this as the end result, successful marginalization over 'v' aleatoric uncertainty variable space
    lp_y_g_x = logsumexp(lp_y_g_xv, axis=1, keepdims=True) - jnp.log(n_v)


def test_vect_choice():
    k = rand.key(47)
    keys = rand.split(k, 3)
    ps = jnp.array([[0.1, 0.3, 0.2, 0.05, 0.35], [0.1, 0.3, 0.2, 0.05, 0.35], [0.1, 0.3, 0.2, 0.05, 0.35]])
    items = jnp.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]])
    choice_vect = jax.vmap(rand.choice, (0, 0, None, None, 0), 0)
    s = choice_vect(keys, items, (5,), True, ps)

    a = jnp.array([[1, 2], [3, 4]])
    inds = jnp.array([[1, 0], [0, 0]])
    def reindex(ar, inds):
        return ar[inds]
    b = jax.vmap(reindex)(a, inds)
    print(b)


if __name__ == '__main__':
    test_vect_choice()
