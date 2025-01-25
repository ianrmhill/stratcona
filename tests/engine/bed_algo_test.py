# Copyright (c) 2025 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
import jax.random as rand
from jax.scipy.special import logsumexp
import numpyro.distributions as dists

import stratcona


def test_bed_algo_log_weight_computation():
    # Generate some observed data
    u, v = 2.3, 1.0
    d = dists.Normal(u, v)
    k = rand.key(826749)
    k1, k2, k3 = rand.split(k, 3)
    n_dev = 5
    measd = d.sample(k1, (1, n_dev, 1, 1))

    # Define a simple probabilistic model
    mb = stratcona.SPMBuilder(mdl_name='hierarchical_test')
    mb.add_hyperlatent('u', dists.Normal, {'loc': 2.2, 'scale': 0.2})
    mb.add_hyperlatent('v', dists.Normal, {'loc': 1.2, 'scale': 0.2}, transform=dists.transforms.SoftplusTransform())
    mb.add_latent('y', nom='u', dev='v')
    mb.add_params(obs_var=0.2)
    mb.add_observed('y_obs', dists.Normal, {'loc': 'y', 'scale': 'obs_var'}, n_dev)
    am = stratcona.AnalysisManager(mb.build_model(), rng_seed=843764)
    tst = stratcona.ReliabilityTest({'t1': {'lot': 1, 'chp': 1}}, {'t1': {}})
    am.set_test_definition(tst)

    # Sampling of x and v values
    n_x, n_v, n_y = 3, 4, 1
    x_s = am.relmdl.sample(k2, am.test, (n_x,), keep_sites=['u', 'v'])
    x_s_tiled = {x: jnp.repeat(jnp.expand_dims(x_s[x], 1), n_v, axis=1) for x in x_s}
    v_s = am.relmdl.sample(k3, am.test, num_samples=(n_x, n_v), keep_sites=am.relmdl._ltnt_subsamples, conditionals=x_s_tiled)

    # All the dimensions need to be correctly arranged for each element to be handled correctly
    x_s_tiled_2 = {x: jnp.repeat(jnp.expand_dims(x_s_tiled[x], axis=2), n_y, axis=2) for x in x_s_tiled}
    v_s_tiled = {v: jnp.repeat(jnp.expand_dims(v_s[v], axis=2), n_y, axis=2) for v in v_s}
    y_tiled = jnp.repeat(jnp.repeat(jnp.expand_dims(measd, axis=(0, 1)), n_v, axis=1), n_x, axis=0)

    lp_y_g_xv = am.relmdl.logp(k1, am.test, site_vals={'t1_y_obs': y_tiled}, conditional=x_s_tiled_2 | v_s_tiled, dims=(n_x, n_v, n_y))

    lp_y_g_x = logsumexp(lp_y_g_xv, axis=1, keepdims=True) - jnp.log(n_v)
    lp_y = logsumexp(lp_y_g_x, axis=0, keepdims=True) - jnp.log(n_x)

    lw_z = lp_y_g_x - lp_y

    assert jnp.all(jnp.equal(lw_z, jnp.array([0.0, 1.2])))


def test_marginalization_over_v_result():
    # Generate some observed data
    u, v1, v2, v3 = 2.3, 1.0, 0.5, 0.7
    d = dists.Normal(u, v1)
    d_v2 = dists.Normal(0, v2)
    d_v3 = dists.Normal(0, v3)
    k = rand.key(557393)
    k1, k2, k3, k4, k5 = rand.split(k, 5)
    n_dev, n_chp, n_lot = 5, 4, 3
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

    # Sampling of x and v values
    n_x, n_v, n_y = 3, 4, 1
    x_s = am.relmdl.sample(k4, am.test, (n_x,), keep_sites=['u', 'v1', 'v2', 'v3'])
    x_s_tiled = {x: jnp.repeat(jnp.expand_dims(x_s[x], 1), n_v, axis=1) for x in x_s}
    v_s = am.relmdl.sample(k5, am.test, num_samples=(n_x, n_v), keep_sites=am.relmdl._ltnt_subsamples, conditionals=x_s_tiled)

    # All the dimensions need to be correctly arranged for each element to be handled correctly
    x_s_tiled_2 = {x: jnp.repeat(jnp.expand_dims(x_s_tiled[x], axis=2), n_y, axis=2) for x in x_s_tiled}
    v_s_tiled = {v: jnp.repeat(jnp.expand_dims(v_s[v], axis=2), n_y, axis=2) for v in v_s}
    y_tiled = jnp.repeat(jnp.repeat(jnp.expand_dims(measd, axis=(0, 1)), n_v, axis=1), n_x, axis=0)

    lp_y_g_xv = am.relmdl.logp(k1, am.test, site_vals={'t1_y_obs': y_tiled}, conditional=x_s_tiled_2 | v_s_tiled, dims=(n_x, n_v, n_y))

    lp_y_g_x = logsumexp(lp_y_g_xv, axis=1, keepdims=True) - jnp.log(n_v)

    x_s_tiled_no_v = {x: jnp.repeat(jnp.expand_dims(x_s[x], axis=1), n_y, axis=1) for x in x_s}
    v_shape = am.relmdl.sample(k5, am.test, num_samples=(n_x, 1), keep_sites=am.relmdl._ltnt_subsamples)
    tiled_zeros = {v: jnp.zeros_like(v_shape[v]) for v in v_shape}
    y_s_tiled_no_v = jnp.repeat(jnp.expand_dims(measd, axis=0), n_x, axis=0)
    lp_y_g_x_shortcut = am.relmdl.logp(k1, am.test, site_vals={'t1_y_obs': y_s_tiled_no_v},
                                       conditional=x_s_tiled_no_v | tiled_zeros, dims=(n_x, n_y))


def test_info_gain_computation():
    # Generate some observed data
    u, v = 2.3, 1.0
    d = dists.Normal(u, v)
    k = rand.key(74843)
    k1, k2, k3 = rand.split(k, 3)
    n_dev = 5
    measd = d.sample(k1, (1, n_dev, 1, 1))

    var_tf = dists.transforms.ComposeTransform([dists.transforms.SoftplusTransform(), dists.transforms.AffineTransform(0, 0.1)])
    # Define a simple probabilistic model
    mb = stratcona.SPMBuilder(mdl_name='hierarchical_test')
    mb.add_hyperlatent('u', dists.Normal, {'loc': 2.2, 'scale': 0.2})
    mb.add_hyperlatent('v', dists.Normal, {'loc': 12, 'scale': 2}, transform=var_tf)
    mb.add_latent('y', nom='u', dev='v')
    mb.add_params(obs_var=0.2)
    mb.add_observed('y_obs', dists.Normal, {'loc': 'y', 'scale': 'obs_var'}, n_dev)
    am = stratcona.AnalysisManager(mb.build_model(), rng_seed=843764)
    tst = stratcona.ReliabilityTest({'t1': {'lot': 1, 'chp': 1}}, {'t1': {}})
    am.set_test_definition(tst)

    # Sampling of x and v values
    n_x, n_v, n_y = 30, 40, 1
    x_s = am.relmdl.sample(k2, am.test, (n_x,), keep_sites=['u', 'v'])
    x_s_tiled = {x: jnp.repeat(jnp.expand_dims(x_s[x], 1), n_v, axis=1) for x in x_s}
    v_s = am.relmdl.sample(k3, am.test, num_samples=(n_x, n_v), keep_sites=am.relmdl._ltnt_subsamples, conditionals=x_s_tiled)

    # All the dimensions need to be correctly arranged for each element to be handled correctly
    x_s_tiled_2 = {x: jnp.repeat(jnp.expand_dims(x_s_tiled[x], axis=2), n_y, axis=2) for x in x_s_tiled}
    v_s_tiled = {v: jnp.repeat(jnp.expand_dims(v_s[v], axis=2), n_y, axis=2) for v in v_s}
    y_tiled = jnp.repeat(jnp.repeat(jnp.expand_dims(measd, axis=(0, 1)), n_v, axis=1), n_x, axis=0)

    lp_y_g_xv = am.relmdl.logp(k1, am.test, site_vals={'t1_y_obs': y_tiled}, conditional=x_s_tiled_2 | v_s_tiled, dims=(n_x, n_v, n_y))
    lp_x = am.relmdl.logp(k1, am.test, site_vals=x_s, conditional=None, dims=(n_x,))

    lp_y_g_x = logsumexp(lp_y_g_xv, axis=1, keepdims=True) - jnp.log(n_v)
    lp_y = logsumexp(lp_y_g_x, axis=0, keepdims=True) - jnp.log(n_x)

    lw_z = lp_y_g_x - lp_y
    lw_z_norm = logsumexp(lw_z, axis=0)
    w_z_norm = jnp.exp(lw_z_norm)
    w_z = jnp.exp(lw_z)

    # Now compute metrics (summary statistics) based on the importance weights
    lp_x_tiled = jnp.expand_dims(lp_x, axis=(1, 2))
    info = lp_y - lp_x_tiled - lp_y_g_x
    entropy_x_g_y = jnp.sum(w_z * info) / w_z_norm
    entropy_x = jnp.sum(-lp_x) / n_x
    info_gain = entropy_x - entropy_x_g_y
    print(info_gain)
