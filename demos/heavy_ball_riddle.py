# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import jax.random as rand
import numpyro.distributions as dists
from numpyro.handlers import trace, seed

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

    #mb.add_hyperlatent('i_heavy_ball', dists.Categorical, {'probs': jnp.full((8,), 0.125)})
    mb.add_hyperlatent('i_heavy_ball', dists.Categorical, {'probs': jnp.array([0.32, 0.32, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06])})

    mb.add_dependent('scale_pos', scale_weigh)
    mb.add_params(outcome_var=0.03)
    mb.add_measured('outcome', dists.Normal, {'loc': 'scale_pos', 'scale': 'outcome_var'}, 1)

    am = stratcona.AnalysisManager(mb.build_model(), rng_seed=92764927)

    test = stratcona.ReliabilityTest({'t1': {'lot': 1, 'chp': 1}}, {'t1': {'placements': jnp.array([0, 1, 2, 2, 2, 2, 2, 2])}})
    am.set_test_definition(test)
    res = am.sim_test_measurements(rtrn_tr=True)
    print(res['t1_outcome'])

    ### Determine Best Experiment Step ###
    #rng_key = rand.key(89)

    #ltnt_samples = am.relmdl.sample(rng_key, test, num_samples=3, keep_sites=am.relmdl.hyls + am.relmdl.ltnts)
    #ltnt_lps = am.relmdl.logp(rng_key, test, ltnt_samples)

    #obs_sample = am.relmdl.sample(rng_key, test, num_samples=1, keep_sites=[f't1_outcome'])
    #obs_samples = {'t1_outcome': jnp.repeat(obs_sample['t1_outcome'], 3, axis=0)}

    #obs_lps = am.relmdl.logp(rng_key, test, obs_samples, ltnt_samples)
    #print(obs_lps)

    #obs_samples = am.relmdl.sample(rng_key, test, num_samples=3, keep_sites=[f't1_outcome'])
    #obs_samples = {'t1_outcome': jnp.repeat(jnp.expand_dims(obs_samples['t1_outcome'], 0), 3, axis=0)}
    #ltnt_samples = am.relmdl.sample_new(rng_key, test, num_samples=(3, 3), keep_sites=am.relmdl.hyls + am.relmdl.ltnts)
    #obs_lp = am.relmdl.logp_new(rng_key, test, obs_samples, ltnt_samples | obs_samples)

    #ltnt_sampler = stratcona.assistants.iterator.iter_sampler([[[0]], [[1]], [[2]], [[3]], [[4]], [[5]], [[6]], [[7]]])
    #obs_sampler = stratcona.assistants.iterator.iter_sampler([jnp.array([[[-1]]]), jnp.array([[[0]]]), jnp.array([[[1]]])])

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

    exp_sampler = stratcona.assistants.iterator.iter_sampler([
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

    # RIG threshold to ensure that the test narrows it down to three balls is:
    # <entropy of 1 in 8> - <entropy of 1 in 3> = 3 bits - 1.5849625 bits = 1.4150375 bits = 0.980829253 nats

    # EIG of one per side: 1.061278, two: 1.5, three: 1.561278, four: 1.0 (bits)

    ### Simulate the Experiment Step ###
    observed_pos = -1

    ### Inference Step ###
    #am.set_experiment_conditions({'single': {'b0p': 0, 'b1p': 0, 'b2p': 0, 'b3p': 1, 'b4p': 1, 'b5p': 1, 'b6p': 2, 'b7p': 2}})
    #tm.infer_model({'single': {'scale_pos': observed_pos}})


if __name__ == '__main__':
    riddle_demo()
