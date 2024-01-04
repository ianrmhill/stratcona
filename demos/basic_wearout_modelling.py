# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pymc

import os
import sys
# This line adds the parent directory to the module search path so that the Stratcona module can be seen and imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stratcona.modelling.builder import ModelBuilder # noqa: ImportNotAtTopOfFile
from stratcona.assistants.probability import shorthand_compile # noqa: ImportNotAtTopOfFile
from stratcona.engine.boed import boed_runner # noqa: ImportNotAtTopOfFile


def wearout_boed():
    # Model provided in JEDEC's JEP122H as generally used NBTI degradation model, equation 5.3.1
    boltz_ev = 8.617e-5
    def bti_vth_shift_empirical(a0, e_aa, temp, vdd, alpha, time, n):
        return a0 * np.exp(e_aa / (boltz_ev * temp)) * (vdd ** alpha) * (time ** n)
    # TODO: Add HCI, TDDB, and EM mechanisms to affect overall performance of ring oscillator circuit. That will be the
    #       target example for now

    # 1. Define the inference model
    builder = ModelBuilder(mdl_name='BTI Degradation Model')
    builder.add_latent_variable('a0', pymc.Normal, {'mu': 0.006, 'sigma': 0.002})
    builder.add_latent_variable('e_aa', pymc.Normal, {'mu': -0.05, 'sigma': 0.02})
    builder.add_latent_variable('alpha', pymc.Normal, {'mu': 9.5, 'sigma': 0.3})
    builder.add_latent_variable('n', pymc.Normal, {'mu': 0.4, 'sigma': 0.1})

    builder.define_experiment_params(['temp', 'vdd', 'time'], simultaneous_experiments=2, samples_per_experiment=10)

    # TODO: We will use the predicted lifespan confidence as our optimization target and basis for latent normalization
    failure_threshold, field_temp, field_vdd = 0.015, 300, 0.9
    def predict_lifespan(a0, e_aa, alpha, n):
        # Computes the time required to reach a given deltaVth threshold under the determined field use conditions
        val = (1/a0) * np.exp(-e_aa / (boltz_ev * field_temp)) * (field_vdd ** -alpha)
        return np.emath.logn(n, val * failure_threshold)

    #builder.add_dependent_variable('FieldLifespan', predict_lifespan)
    builder.add_dependent_variable('Delta_Vth', bti_vth_shift_empirical)
    builder.set_variable_observed('Delta_Vth', variability=0.02)

    mdl, ltnt_list, obs_list = builder.build_model(ltnt_normalization=None)

    # 2. Create the necessary samplers and probability compute graphs
    # Define a proposal for the marginal probability (the distribution of vth shift across all tests and latent values)
    with pymc.Model() as marg_proposal:
        y_int = pymc.Normal('vth_shift_expected', [0.01, 0.008], [0.01, 0.008], shape=(10, 2))
    y_sampler = pymc.compile_pymc([], y_int, name='marg_sampler')
    # Compile the model samplers needed for BOED
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

    # 4. Run Gerabaldi simulation to emulate running the test and getting observations
    # TODO: Run simulation
    builder.observed_handle.set_value(np.array([[[0.02, 0.024], [0.021, 0.021], [0.021, 0.018], [0.021, 0.021],
                                                 [0.021, 0.021], [0.02, 0.021], [0.021, 0.021], [0.024, 0.026],
                                                 [0.017, 0.022], [0.021, 0.021]]]))

    # 5. Infer the posterior for the model based on the observed data
    idata = pymc.sample(model=mdl)

    # 6. Repeat and visualize results


if __name__ == '__main__':
    wearout_boed()
