# Copyright (c) 2025 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import numpyro.distributions as dist

import jax
import jax.numpy as jnp
from functools import partial
import pytest

from stratcona.modelling.relmodel import TestDef, ReliabilityModel, ReliabilityRequirement, ExpDims
from stratcona.modelling.builder import SPMBuilder
from stratcona.manager import AnalysisManager


def test_relreq():
    def faux_metric(s):
        return jnp.min(s)

    req = ReliabilityRequirement(faux_metric, 0.8, 20)

    assert req.type == faux_metric
    assert req.quantile == 0.8
    assert req.target_lifespan == 20


def test_expdims():
    # Test the basic functionality
    devs = {'e1': 6, 'e2': 7, 'e3': 9}
    dims = ExpDims('eg', 4, 5, **devs)
    assert dims.name == 'eg'
    assert dims.lot == 4
    assert dims.chp == 5
    assert dims.dev['e3'] == 9

    # Test hash and eq special method implementations
    devs2 = {'e2': 7, 'e1': 6, 'e3': 9}
    dims2 = ExpDims('eg', 4, 5, **devs2)
    with pytest.raises(TypeError):
        dims2.dev['e1'] = 2
    assert dims == dims2
    assert hash(dims) == hash(dims2)
    devs3 = {'e2': 7, 'e3': 6, 'e1': 9}
    dims3 = ExpDims('eg', 4, 5, **devs3)
    assert dims != dims3
    assert hash(dims) != hash(dims3)


def test_testdef():
    # Test the basic functionality
    td = TestDef('mytest', {'e1': {'lot': 2, 'chp': 3}, 'e2': {'lot': 4, 'chp': 5}},
                 {'e1': {'temp': 4.3}, 'e2': {'temp': 4.5}})
    assert td.name == 'mytest'
    assert td.conds['e2']['temp'] == 4.5
    assert td.conds['e1']['temp'] == 4.3

    td2 = TestDef('mytest', {'e2': {'lot': 4, 'chp': 5}, 'e1': {'lot': 2, 'chp': 3}},
                  {'e2': {'temp': 4.5}, 'e1': {'temp': 4.3}})
    assert td != td2
    td2 = TestDef('mytest', {'e1': {'lot': 2, 'chp': 3}, 'e2': {'lot': 4, 'chp': 5}},
                  {'e2': {'temp': 4.5}, 'e1': {'temp': 4.3}})
    assert td == td2


def test_relmdl():
    mb = SPMBuilder(mdl_name='barebones')
    mb.add_hyperlatent('x', dist.Normal, {'loc': 0, 'scale': 0.0001})
    mb.add_params(eps=0.2)
    mb.add_observed('y', dist.Normal, {'loc': 'x', 'scale': 'eps'}, 5)
    am = AnalysisManager(mb.build_model(), rng_seed=911)

    # Test basic members of the model
    assert am.relmdl.name == 'barebones'
    assert am.relmdl.param_vals == {'eps': 0.2}
    assert am.relmdl.hyl_beliefs == {'x': {'loc': 0, 'scale': 0.0001}}
    assert am.relmdl.hyls == ('x',)
    assert am.relmdl.hyl_info['x']['dist'] == dist.Normal
    assert am.relmdl.ltnts == ()
    assert am.relmdl.ltnt_subsamples == ()
    assert am.relmdl.observes == ('y',)
    assert am.relmdl.obs_per_chp == {'y': 5}
    assert am.relmdl.obs_noise == {'y': 0.2}
    assert am.relmdl.predictors == ()
    assert am.relmdl.fail_criteria == ()
    assert am.relmdl.i_s_override is None
    assert am.relmdl.y_s_override is None

    # Test the basics of the sample and logprob methods
    lpk, dims, conds = am._derive_key(), (ExpDims('e', 1, 1),), {'e': {}}
    s = am.relmdl.sample(lpk, dims, conds, (2, 1))
    lp = am.relmdl.logprob(lpk, dims, conds, {'e_y': s['e_y']}, None, (2, 1))
    assert jnp.allclose(jnp.round(s['x'], 5), jnp.array([[-0.00013], [-0.00003]]))
    assert jnp.allclose(jnp.round(s['e_y'], 3),
                        jnp.array([[[[[-0.087]], [[-0.183]], [[-0.108]], [[-0.141]], [[-0.047]]]],
                                   [[[[0.266]], [[-0.014]], [[-0.241]], [[0.268]], [[0.037]]]]]))
    assert jnp.allclose(jnp.round(lp, 3), jnp.array([[2.523], [0.925]]))

    # Test the setter logic that clears out jax caches when model values are changed
    sn = dist.Normal(loc=0.2, scale=0.2).sample(am._derive_key(), (5, 1, 1))
    lp1 = am.relmdl.logprob(lpk, dims, conds, {'e_y': sn}, None)
    am.relmdl.hyl_beliefs = {'x': {'loc': 0.2, 'scale': 0.0001}}
    lp2 = am.relmdl.logprob(lpk, dims, conds, {'e_y': sn}, None)
    assert lp1 != lp2
