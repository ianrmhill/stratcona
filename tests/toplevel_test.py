# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytensor as pt
import pymc

from stratcona.assistants.iterator import *
from stratcona.assistants.probability import *
from stratcona.modelling.variables import *
from stratcona.modelling.builder import *
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
    ltnt_logp = shorthand_compile('ltnt_logp', riddle_mdl, [i_heavy_ball], [output], [])
    obs_sampler = iter_sampler([-1, 0, 1])
    obs_logp = shorthand_compile('obs_logp', riddle_mdl, [i_heavy_ball], [output], [])

    exp_sampler = iter_sampler([[0, 1, 2, 2, 2, 2, 2, 2], [0, 0, 1, 1, 2, 2, 2, 2],
                                [0, 0, 0, 1, 1, 1, 2, 2], [0, 0, 0, 0, 1, 1, 1, 1]])

    results = boed_runner(4, 3, 8, exp_sampler, exp_handle, ltnt_sampler, obs_sampler, ltnt_logp, obs_logp)
    # Convert EIG to bits before comparison for readability
    nats_to_bits = np.log2(np.e)
    results['eig'] = results['eig'] * nats_to_bits
    assert results['eig'].round(5).tolist() == [1.06128, 1.5, 1.56128, 1.0]


def test_bti_model():
    # Model provided in JEDEC's JEP122H as generally used NBTI degradation model, equation 5.3.1
    boltz_ev = 8.617e-5
    def bti_vth_shift_empirical(a0, e_aa, temp, vdd, alpha, time, n):
        return a0 * np.exp(e_aa / (boltz_ev * temp)) * (vdd ** alpha) * (time ** n)
        # return 100 * (0.001 * a_0) * np.exp((0.01 * e_aa) / (boltz_ev * temp)) * (vdd ** alpha) * (time ** (0.1 * n))

    # 1. Define the inference model
    builder = ModelBuilder(mdl_name='BTI Degradation Model')
    builder.add_latent_variable('a0', pymc.Normal, {'mu': 6, 'sigma': 2})
    builder.add_latent_variable('e_aa', pymc.Normal, {'mu': -5, 'sigma': 2})
    builder.add_latent_variable('alpha', pymc.Normal, {'mu': 9.5, 'sigma': 0.3})
    builder.add_latent_variable('n', pymc.Normal, {'mu': 4, 'sigma': 1})

    builder.define_experiment_params(['temp', 'vdd', 'time'], simultaneous_experiments=2, samples_per_experiment=10)

    builder.add_dependent_variable('Delta_Vth', bti_vth_shift_empirical)
    builder.set_variable_observed('Delta_Vth', variability=0.02)

    mdl, ltnt_list, obs_list = builder.build_model()

    # 2. Create the necessary samplers and probability compute graphs
    with pymc.Model() as marg_proposal:
        y_int = pymc.Normal('vth_shift_expected', [1, 0.1], [1, 0.1], shape=(10, 2))
    y_sampler = pymc.compile_pymc([], y_int, name='marg_sampler')
    prior_sampler = shorthand_compile('ltnt_sampler', mdl, ltnt_list, obs_list)
    ltnt_logp = shorthand_compile('ltnt_logp', mdl, ltnt_list, obs_list)
    obs_logp = shorthand_compile('obs_logp', mdl, ltnt_list, obs_list)

    # 3. Determine the best experiment to conduct
    def exp_sampler():
        temps_possible = [400, 410, 420, 430, 440]
        vdds_possible = [0.8, 0.88, 0.96, 1]
        times_possible = [100, 300, 500, 800, 1000]
        return [[np.random.choice(temps_possible), np.random.choice(vdds_possible), np.random.choice(times_possible)],
                [np.random.choice(temps_possible), np.random.choice(vdds_possible), np.random.choice(times_possible)]]

    best_test = boed_runner(30, 100, 100, exp_sampler, builder.experiment_handle,
                            prior_sampler, y_sampler, ltnt_logp, obs_logp, return_best_only=True)
    print(f"Best test: {best_test[0]} with EIG: {best_test[1]} nats")
