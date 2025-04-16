# Copyright (c) 2025 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import jax
import numpyro.distributions as dist

import jax.numpy as jnp
import numpy as np

import time as time

import gerabaldi
from gerabaldi.models import *

import os
import sys
# This line adds the parent directory to the module search path so that the Stratcona module can be seen and imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# TEMP
from matplotlib import pyplot as plt

import stratcona


def htol_demo():
    jax.config.update('jax_enable_compilation_cache', False)
    ### Define some constants ###
    boltz_ev = 8.617e-5
    # 125C in Kelvin
    htol_temp = 398.15

    ########################################################################
    ### 1. Define the predictive wear-out model to infer                 ###
    ########################################################################
    # Define the model we will use to fit degradation
    mb = stratcona.SPMBuilder(mdl_name='Arrhenius')

    # Generic Arrhenius model used by JEDEC for abstract-level lifespan extrapolation from accelerated conditions
    def arrhenius_t_and_v(a, eaat, eaav, temp, volt):
        temp_coeff = jnp.exp((-0.1 * eaat) / (boltz_ev * temp)) * 1e9
        volt_coeff = jnp.exp((-0.1 * eaav) / (boltz_ev * volt * 1000)) * 1e9
        return 1e9 * a * temp_coeff * volt_coeff

    print(arrhenius_t_and_v(4, 7, 30, 330, 0.85))
    print(arrhenius_t_and_v(4, 7, 30, 330, 0.9))
    print(arrhenius_t_and_v(4, 7, 30, 400, 0.95))

    var_tf = dist.transforms.ComposeTransform([dist.transforms.SoftplusTransform(), dist.transforms.AffineTransform(0, 0.01)])
    mb.add_hyperlatent('a_nom', dist.Normal, {'loc': 4, 'scale': 1})
    mb.add_hyperlatent('a_chp', dist.Normal, {'loc': 8, 'scale': 3}, transform=var_tf)
    mb.add_hyperlatent('a_lot', dist.Normal, {'loc': 12, 'scale': 3}, transform=var_tf)
    mb.add_hyperlatent('eaat_nom', dist.Normal, {'loc': 7, 'scale': 0.5})
    mb.add_hyperlatent('eaat_chp', dist.Normal, {'loc': 8, 'scale': 3}, transform=var_tf)
    mb.add_hyperlatent('eaat_lot', dist.Normal, {'loc': 12, 'scale': 3}, transform=var_tf)
    mb.add_hyperlatent('eaav_nom', dist.Normal, {'loc': 5, 'scale': 1})
    mb.add_hyperlatent('eaav_chp', dist.Normal, {'loc': 8, 'scale': 3}, transform=var_tf)
    mb.add_hyperlatent('eaav_lot', dist.Normal, {'loc': 12, 'scale': 3}, transform=var_tf)

    mb.add_latent('a', 'a_nom', chp='a_chp', lot='a_lot')
    mb.add_latent('eaat', 'eaat_nom', chp='eaat_chp', lot='eaat_lot')
    mb.add_latent('eaav', 'eaav_nom', chp='eaav_chp', lot='eaav_lot')

    mb.add_intermediate('fit', arrhenius_t_and_v)
    mb.add_params(fitvar=0.01)
    mb.add_observed('fit_calcd', dist.Normal, {'loc': 'fit', 'scale': 'fitvar'}, 1)

    #def slowest_ro(fmeas, fmin):
    #    return jnp.any(fmeas < fmin)
    #mb.add_fail_criterion('margin_violation', slowest_ro)

    ########################################################################
    am = stratcona.AnalysisManager(mb.build_model(), rng_seed=9283747)
    am.set_field_use_conditions({'volt': 0.85, 'temp': 330})
    tsample = stratcona.TestDef('sample_vals', {'b1': {'lot': 3, 'chp': 7}}, {'b1': {'volt': 0.95, 'temp': 400}})
    am.set_test_definition(tsample)
    fit_samples = am.sim_test_meas_new()

    print(fit_samples)
    return

    #start_time = t.time()
    #estimate = tm.estimate_reliability(num_samples=3000)
    #print(f"Estimated product lifespan: {estimate} hours")
    #print(f"Lifespan prediction time: {t.time() - start_time} seconds")

    #plt.show()

    t1 = stratcona.TestDef(
        'htol', {'b1': {'lot': 1, 'chp': 77}, 'b2': {'lot': 1, 'chp': 77}, 'b3': {'lot': 1, 'chp': 77}},
        {'b1': {'volt': 0.9, 'temp': 400}, 'b2': {'volt': 0.9, 'temp': 400}, 'b3': {'volt': 0.9, 'temp': 400}})
    t2 = stratcona.TestDef(
        'htol', {'b1': {'lot': 1, 'chp': 77}, 'b2': {'lot': 1, 'chp': 77}, 'b3': {'lot': 1, 'chp': 77}},
        {'b1': {'volt': 0.9, 'temp': 350}, 'b2': {'volt': 0.9, 'temp': 400}, 'b3': {'volt': 0.9, 'temp': 425}})
    possible_tests = [t1, t2]
    exp_sampler = stratcona.assistants.iter_sampler(possible_tests)

    ########################################################################
    ### 3. Determine the best experiment to conduct                      ###
    ########################################################################
    start_time = time.time()
    eigs, pstats = am.determine_best_test_apr25(2, 10, 10, 10, exp_sampler)
    print(f"Test EIG estimation time: {time.time() - start_time} seconds")
    print(eigs)

    return

    ########################################################################
    ### 4. Simulate the Experiment                                       ###
    ########################################################################
    to_meas = MeasSpec({'deg': num_devices}, {'temp': 300, 'vdd': 0.85}, 'Measure Ten')
    # We will run two tests, one with high EIG, one with poor EIG
    best_strs = StrsSpec({'temp': 300, 'vdd': 0.84}, 500, 'Best Vdd')
    poor_strs = StrsSpec({'temp': 300, 'vdd': 1.15}, 500, 'Poor Vdd')
    best_test = TestSpec([best_strs, to_meas], num_chps=1, num_lots=1, name='Best Test')
    poor_test = TestSpec([poor_strs, to_meas], num_chps=1, num_lots=1, name='Poor Test')

    test_env = PhysTestEnv(env_vrtns={
        'temp': EnvVrtnMdl(dev_vrtn_mdl=Normal(0, 0.1), chp_vrtn_mdl=Normal(0, 0.4)),
        'vdd': EnvVrtnMdl(dev_vrtn_mdl=Normal(0, 0.003))
    })

    # The actual model is not of the same form as the model used for reliability prediction (here a power term is added)
    #def threshold_degradation(a, b, c, d, vdd, temp, time):
    #    return np.where(np.greater(vdd, a), (b * (temp - (c * 100)) * (time / 1000)) ** d, 0)

    def threshold_degradation(a, c, vdd, temp, time):
        #d = np.where(np.greater(vdd, a), b * (temp - (c * 100)) * (time / 1000), 0)
        d = np.where(np.greater(vdd, a), (temp - (c * 100)) * (time / 1000), 0)
        return d

    dev_mdl = DeviceMdl({'deg': DegPrmMdl(DegMechMdl(
        threshold_degradation, mdl_name='Threshold',
        a=LatentVar(deter_val=0.84),
        #b=LatentVar(deter_val=0.39), c=LatentVar(deter_val=1.48), d=LatentVar(deter_val=0.96)))})
        c=LatentVar(deter_val=1.48)))})

    start_time = t.time()
    best_test_rslts = gerabaldi.simulate(best_test, dev_mdl, test_env)
    poor_test_rslts = gerabaldi.simulate(poor_test, dev_mdl, test_env)
    print(f"Simulation time: {t.time() - start_time} seconds")

    # Extract the 10 raw values from the test results
    best_vals = best_test_rslts.measurements['measured'].to_numpy()
    poor_vals = poor_test_rslts.measurements['measured'].to_numpy()
    print(f"Best test simulation measured: {best_vals}.")
    print(f"Poor test simulation measured: {poor_vals}.")

    ### Inference Step for Each Test ###
    priors = tm.get_priors(for_user=True)
    tm.set_experiment_conditions({'htol': {'vdd': 0.84, 'temp': 300, 'time': 500}})
    tm.infer_model({'htol': {'delta_vth': best_vals}})

    # Reset model for second inference
    tm.set_priors(priors)
    tm.set_experiment_conditions({'htol': {'vdd': 1.15, 'temp': 300, 'time': 500}})
    tm.infer_model({'htol': {'delta_vth': poor_vals}})

    new_estimate = tm.estimate_reliability()
    print(new_estimate)


if __name__ == '__main__':
    htol_demo()
