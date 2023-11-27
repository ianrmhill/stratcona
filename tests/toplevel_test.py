# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytensor as pt
import pymc

from stratcona.assistants.iterator import *
from stratcona.assistants.probability import *
from stratcona.engine.boed import boed_runner


def test_heavy_ball():
    # This test is based on an experimental design riddle in which 8 balls are given of which one is imperceptibly
    # heavier. A weighing scale (see-saw style) is given with which you must figure out how to exactly identify the
    # heavier ball with only two weighings of the scale.

    # All balls have equal probability of being heavy
    ball_priors = np.array([0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125])

    theta_handle = pt.shared(ball_priors)
    exp_handle = pt.shared(np.array([0, 0, 0, 1, 1, 1, 2, 2]))
    weights = pt.shared(np.ones(8))
    with pymc.Model() as riddle_mdl:
        # Get a sample of which ball is the heavy one
        i_heavy_ball = pymc.Categorical('which_heavy', theta_handle)
        adj_weights = pt.tensor.subtensor.set_subtensor(weights[i_heavy_ball], 1.1)

        w_one = pt.tensor.sum(pt.tensor.where(pt.tensor.eq(exp_handle, 0), 1, 0) * adj_weights)
        w_two = pt.tensor.sum(pt.tensor.where(pt.tensor.eq(exp_handle, 1), 1, 0) * adj_weights)
        scale_state = pt.tensor.where(w_one > w_two, 1, 0)
        scale_state = pt.tensor.where(w_one < w_two, -1, scale_state)

        output = pymc.Normal('result', scale_state, 0.01)

    # Compile the model functions needed to compute expected information gain and create iterators over the variable
    # support ranges
    ltnt_sampler = iter_sampler([0, 1, 2, 3, 4, 5, 6, 7])
    ltnt_logp = shorthand_compile('ltnt_logp', riddle_mdl, [i_heavy_ball], [output])
    obs_sampler = iter_sampler([-1, 0, 1])
    obs_logp = shorthand_compile('obs_logp', riddle_mdl, [i_heavy_ball], [output])

    exp_sampler = iter_sampler([[0, 1, 2, 2, 2, 2, 2, 2], [0, 0, 1, 1, 2, 2, 2, 2],
                                [0, 0, 0, 1, 1, 1, 2, 2], [0, 0, 0, 0, 1, 1, 1, 1]])

    results = boed_runner(4, 3, 8, exp_sampler, exp_handle, ltnt_sampler, obs_sampler, ltnt_logp, obs_logp)
    # Convert EIG to bits before comparison for readability
    nats_to_bits = np.log2(np.e)
    results['eig'] = results['eig'] * nats_to_bits
    assert results['eig'].round(5).tolist() == [1.06128, 1.5, 1.56128, 1.0]
