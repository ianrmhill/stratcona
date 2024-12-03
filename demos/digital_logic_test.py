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


def logic_test_demo():
    mb = stratcona.SPMBuilder(mdl_name='Logic Circuit with Possible Faults')

    def run_circuit(n1_f, n2_f, n3_f, n4_f, n5_f, inputs):
        n1 = jnp.where(n1_f == 2, inputs[0], n1_f)
        n2 = jnp.where(n2_f == 2, inputs[1], n2_f)
        n3 = jnp.where(n3_f == 2, inputs[2], n3_f)
        n4 = jnp.where(n4_f == 2, n1 | n2, n4_f)
        n5 = jnp.where(n5_f == 2, n3 & n4, n5_f)
        return n5

    def is_faulty(n1_f, n2_f, n3_f, n4_f, n5_f):
        faulty = jnp.where(n1_f != 2, True, False)
        faulty = jnp.where(n2_f != 2, True, faulty)
        faulty = jnp.where(n3_f != 2, True, faulty)
        faulty = jnp.where(n4_f != 2, True, faulty)
        faulty = jnp.where(n5_f != 2, True, faulty)
        return faulty

    mb.add_hyperlatent('n1_f', dists.Categorical, {'probs': jnp.array([0.05, 0.05, 0.9])})
    mb.add_hyperlatent('n2_f', dists.Categorical, {'probs': jnp.array([0.05, 0.05, 0.9])})
    mb.add_hyperlatent('n3_f', dists.Categorical, {'probs': jnp.array([0.05, 0.05, 0.9])})
    mb.add_hyperlatent('n4_f', dists.Categorical, {'probs': jnp.array([0.05, 0.05, 0.9])})
    mb.add_hyperlatent('n5_f', dists.Categorical, {'probs': jnp.array([0.05, 0.05, 0.9])})

    mb.add_dependent('o', run_circuit)
    mb.add_predictor('faulty', is_faulty, {})
    mb.add_params(o_var=0.03)
    mb.add_measured('v_meas', dists.Normal, {'loc': 'o', 'scale': 'o_var'}, 1)

    am = stratcona.AnalysisManager(mb.build_model(), rng_seed=48939837832)

    test = stratcona.ReliabilityTest({'t1': {'lot': 1, 'chp': 1}}, {'t1': {'inputs': jnp.array([0, 1, 1])}})
    am.set_test_definition(test)

    res = am.sim_test_measurements(rtrn_tr=True)
    print(res['t1_v_meas'])

    ### Determine Best Experiment Step ###

    am.relmdl.y_s_override = True
    def y_s_scale(d, num_samples):
        return {'t1_v_meas': jnp.array([[[[0]]], [[[1]]], [[[0]]], [[[1]]], [[[0]]], [[[1]]], [[[0]]], [[[1]]]], dtype=jnp.float32),
                't2_v_meas': jnp.array([[[[0]]], [[[0]]], [[[1]]], [[[1]]], [[[0]]], [[[0]]], [[[1]]], [[[1]]]], dtype=jnp.float32),
                't3_v_meas': jnp.array([[[[0]]], [[[0]]], [[[0]]], [[[0]]], [[[1]]], [[[1]]], [[[1]]], [[[1]]]], dtype=jnp.float32)}
    am.relmdl.y_s_custom = y_s_scale

    am.relmdl.i_s_override = True
    def i_s_ball_pos(d, num_samples):
        fault_states = jnp.array([0, 1, 2])
        n1 = jnp.tile(fault_states, 3*3*3*3)
        n2 = jnp.tile(jnp.repeat(fault_states, 3), 3*3*3)
        n3 = jnp.tile(jnp.repeat(fault_states, 3*3), 3*3)
        n4 = jnp.tile(jnp.repeat(fault_states, 3*3*3), 3)
        n5 = jnp.repeat(fault_states, 3*3*3*3)

        n1 = jnp.repeat(jnp.expand_dims(n1, 0), 8, axis=0)
        n2 = jnp.repeat(jnp.expand_dims(n2, 0), 8, axis=0)
        n3 = jnp.repeat(jnp.expand_dims(n3, 0), 8, axis=0)
        n4 = jnp.repeat(jnp.expand_dims(n4, 0), 8, axis=0)
        n5 = jnp.repeat(jnp.expand_dims(n5, 0), 8, axis=0)
        return {'n1_f': n1, 'n2_f': n2, 'n3_f': n3, 'n4_f': n4, 'n5_f': n5}
    am.relmdl.i_s_custom = i_s_ball_pos

    tests = [jnp.array([0, 0, 0]), jnp.array([0, 0, 1]), jnp.array([0, 1, 0]), jnp.array([0, 1, 1]),
             jnp.array([1, 0, 0]), jnp.array([1, 0, 1]), jnp.array([1, 1, 0]), jnp.array([1, 1, 1])]
    test_list = []
    for i in range(8):
        for j in range(8):
            for k in range(8):
                test = stratcona.ReliabilityTest(
                    {'t1': {'lot': 1, 'chp': 1}, 't2': {'lot': 1, 'chp': 1}, 't3': {'lot': 1, 'chp': 1}},
                    {'t1': {'inputs': tests[i]}, 't2': {'inputs': tests[j]}, 't3': {'inputs': tests[k]}})
                test_list.append(test)

    exp_sampler = stratcona.assistants.iter_sampler(test_list)

    eigs = am.find_best_experiment(512, 8, 243, exp_sampler)
    eigs_bits = [eigs[i] * 1.442695 for i in range(len(eigs))]
    #eigs_bits = eigs
    eig_max = 0
    test_max = None
    for i, test in enumerate(test_list):
        if eigs_bits[i] > eig_max:
            eig_max = eigs_bits[i]
            test_max = test
    print(f'Max: {test_max.conditions}: {eig_max} bits')
    #print(f'Max: {test_max.conditions}: {eig_max * 100}%')

    ### Simulate the Experiment Step ###
    observed_pos = -1

    ### Inference Step ###
    #am.set_experiment_conditions({'single': {'b0p': 0, 'b1p': 0, 'b2p': 0, 'b3p': 1, 'b4p': 1, 'b5p': 1, 'b6p': 2, 'b7p': 2}})
    #tm.infer_model({'single': {'scale_pos': observed_pos}})


if __name__ == '__main__':
    logic_test_demo()
