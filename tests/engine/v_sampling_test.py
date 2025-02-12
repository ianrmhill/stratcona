# Copyright (c) 2025 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import jax.random as rand
from jax.scipy.special import logsumexp
import numpyro.distributions as dists
import numpy as np

import stratcona
from stratcona.engine.inference import int_out_v
from stratcona.modelling.relmodel import TestDef





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
