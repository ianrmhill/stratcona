# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytensor.tensor as pt
import pytensor
import pymc

# TEMP
from matplotlib import pyplot as plt

import os
import sys
# This line adds the parent directory to the module search path so that the Stratcona module can be seen and imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import stratcona


def riddle_demo():
    mb = stratcona.ModelBuilder(mdl_name='Heavy Ball Riddle')

    def scale_weigh(i_heavy_ball, b0p, b1p, b2p, b3p, b4p, b5p, b6p, b7p):
        positions = pt.as_tensor([b0p, b1p, b2p, b3p, b4p, b5p, b6p, b7p])
        weights = pt.ones_like(positions)
        adj_weights = pt.subtensor.set_subtensor(weights[i_heavy_ball, :, :], 2)

        w_one = pt.sum(pt.where(pt.eq(positions, 0), 1, 0) * adj_weights)
        w_two = pt.sum(pt.where(pt.eq(positions, 1), 1, 0) * adj_weights)
        scale_state = pt.where(w_one > w_two, 1, 0)
        scale_state = pt.where(w_one < w_two, -1, scale_state)
        return scale_state

    mb.add_latent_variable('i_heavy_ball', pymc.Categorical,
                           {'p': [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]})

    mb.define_experiment_params(['b0p', 'b1p', 'b2p', 'b3p', 'b4p', 'b5p', 'b6p', 'b7p'],
                                simultaneous_experiments=['single'],
                                samples_per_experiment={'all': 1})

    mb.add_dependent_variable('scale_pos', scale_weigh)
    mb.set_variable_observed('scale_pos', variability=0.01)

    tm = stratcona.TestDesignManager(mb)

    tm.set_experiment_conditions({'single': {'b0p': 0, 'b1p': 0, 'b2p': 0, 'b3p': 1, 'b4p': 1, 'b5p': 1, 'b6p': 2, 'b7p': 2}})
    tm.set_observations({'scale_pos': -1})
    tm.examine('latents')
    plt.show()

    ### Determine Best Experiment Step ###
    ltnt_sampler = stratcona.assistants.iterator.iter_sampler([[0], [1], [2], [3], [4], [5], [6], [7]])
    tm.override_func('ltnt_sampler', ltnt_sampler)
    obs_sampler = stratcona.assistants.iterator.iter_sampler([np.array([[[-1]]]), np.array([[[0]]]), np.array([[[1]]])])
    #def obs_sampler():
    #    val = np.random.choice([-1, 0, 1])
    #    return np.array([[[val]]])
    tm.override_func('obs_sampler', obs_sampler)
    exp_sampler = stratcona.assistants.iterator.iter_sampler([
        {'b0p': 0, 'b1p': 1, 'b2p': 2, 'b3p': 2, 'b4p': 2, 'b5p': 2, 'b6p': 2, 'b7p': 2},
        {'b0p': 0, 'b1p': 0, 'b2p': 1, 'b3p': 1, 'b4p': 2, 'b5p': 2, 'b6p': 2, 'b7p': 2},
        {'b0p': 0, 'b1p': 0, 'b2p': 0, 'b3p': 1, 'b4p': 1, 'b5p': 1, 'b6p': 2, 'b7p': 2},
        {'b0p': 0, 'b1p': 0, 'b2p': 0, 'b3p': 0, 'b4p': 1, 'b5p': 1, 'b6p': 1, 'b7p': 1}])
    tm.determine_best_test(exp_sampler, None, 4, 3, 8)
    # EIG of one per side: 1.061278, two: 1.5, three: 1.561278, four: 1.0 (bits)

    ### Simulate the Experiment Step ###
    observed_pos = 1

    ### Inference Step ###
    tm.set_experiment_conditions({'single': {'b0p': 0, 'b1p': 0, 'b2p': 0, 'b3p': 1, 'b4p': 1, 'b5p': 1, 'b6p': 2, 'b7p': 2}})
    tm.infer_model({'single': {'scale_pos': observed_pos}})


if __name__ == '__main__':
    riddle_demo()
