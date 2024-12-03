# Copyright (c) 2024 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
import numpyro.distributions as dists

# TEMP
from matplotlib import pyplot as plt

import os
import sys
# This line adds the parent directory to the module search path so that the Stratcona module can be seen and imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import stratcona


def riddle_demo():
    mb = stratcona.SPMBuilder(mdl_name='Heavy Ball Riddle')

    def scale_weigh(i_heavy_ball, placements):
        weights = jnp.ones_like(placements, dtype=jnp.float32)
        adj_weights = weights.at[i_heavy_ball].set(1.2)

        w_one = jnp.sum(jnp.where(placements == 0, 1, 0) * adj_weights)
        w_two = jnp.sum(jnp.where(placements == 1, 1, 0) * adj_weights)
        scale_state = jnp.where(w_one > w_two, 1, 0)
        scale_state = jnp.where(w_one < w_two, -1, scale_state)
        return scale_state

    mb.add_hyperlatent('i_heavy_ball', dists.Categorical, {'probs': jnp.array([0.32, 0.32, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06])})

    mb.add_intermediate('scale_pos', scale_weigh)
    mb.add_params(outcome_var=0.03)
    mb.add_observed('outcome', dists.Normal, {'loc': 'scale_pos', 'scale': 'outcome_var'}, 1)

    am = stratcona.AnalysisManager(mb.build_model(), rng_seed=92764927)

    ### Determine Best Experiment Step ###
    am.relmdl.y_s_override = True
    def y_s_scale(d, num_samples):
        return {'t1_outcome': jnp.array([[[[-1]]], [[[0]]], [[[1]]]], dtype=jnp.float32)}
    am.relmdl.y_s_custom = y_s_scale

    am.relmdl.i_s_override = True
    def i_s_ball_pos(d, num_samples):
        all_positions = jnp.array([0, 1, 2, 3, 4, 5, 6, 7])
        tiled = jnp.repeat(jnp.expand_dims(all_positions, 0), 3, axis=0)
        return {'i_heavy_ball': tiled}
    am.relmdl.i_s_custom = i_s_ball_pos

    exp_sampler = stratcona.assistants.iter_sampler([
        stratcona.ReliabilityTest({'t1': {'lot': 1, 'chp': 1}}, {'t1': {'placements': jnp.array([0, 1, 2, 2, 2, 2, 2, 2])}}),
        stratcona.ReliabilityTest({'t1': {'lot': 1, 'chp': 1}}, {'t1': {'placements': jnp.array([0, 1, 0, 1, 2, 2, 2, 2])}}),
        stratcona.ReliabilityTest({'t1': {'lot': 1, 'chp': 1}}, {'t1': {'placements': jnp.array([0, 2, 0, 1, 1, 2, 2, 2])}}),
        stratcona.ReliabilityTest({'t1': {'lot': 1, 'chp': 1}}, {'t1': {'placements': jnp.array([0, 1, 0, 1, 0, 1, 2, 2])}}),
        stratcona.ReliabilityTest({'t1': {'lot': 1, 'chp': 1}}, {'t1': {'placements': jnp.array([0, 2, 0, 1, 0, 1, 1, 2])}}),
        stratcona.ReliabilityTest({'t1': {'lot': 1, 'chp': 1}}, {'t1': {'placements': jnp.array([2, 2, 0, 0, 0, 1, 1, 1])}}),
        stratcona.ReliabilityTest({'t1': {'lot': 1, 'chp': 1}}, {'t1': {'placements': jnp.array([0, 0, 0, 1, 1, 1, 2, 2])}}),
        stratcona.ReliabilityTest({'t1': {'lot': 1, 'chp': 1}}, {'t1': {'placements': jnp.array([0, 1, 0, 1, 0, 1, 0, 1])}}),
    ])
    eigs = am.find_best_experiment(7, 3, 8, exp_sampler)
    eigs_bits = [eigs[i] * 1.442695 for i in range(len(eigs))]
    print(eigs_bits)

    # EIG of one per side: 1.061278, two: 1.5, three: 1.561278, four: 1.0 (bits)

    ### Simulate the Experiment Step ###
    observed_pos = -1

    ### Inference Step ###
    #am.set_experiment_conditions({'single': {'b0p': 0, 'b1p': 0, 'b2p': 0, 'b3p': 1, 'b4p': 1, 'b5p': 1, 'b6p': 2, 'b7p': 2}})
    #tm.infer_model({'single': {'scale_pos': observed_pos}})


if __name__ == '__main__':
    riddle_demo()
