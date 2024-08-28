# Copyright (c) 2024 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import time as t
import pytensor
import pytensor.tensor as pt
import pymc
from functools import partial
from itertools import product

import jax.numpy as jnp
import jax.random as rand
import numpyro.distributions as dists

import gerabaldi
from gerabaldi.models import *

import os
import sys
# This line adds the parent directory to the module search path so that the Stratcona module can be seen and imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from matplotlib import pyplot as plt

import stratcona

BOLTZ_EV = 8.617e-5

SHOW_PLOTS = True


def electromigration_qualification():
    analyze_prior = True
    run_bed_analysis = True
    run_inference = False
    run_posterior_analysis = False
    simulated_data_mode = 'model'

    mb = stratcona.SPMBuilder(mdl_name='Electromigration')

    '''
    ===== 1) Determine long-term reliability requirements =====
    We will base the requirement on the SIL3 specification of less than 10^-3 (0.001) probability of failure on demand,
    which in the context of long-term reliability we will interpret as a less than 0.1% probability that
    electromigration failures occur before the 10 year intended product useful life. This translates to a metric of a
    worst case 0.1% quantile credible region of at least 10 years.
    '''
    percent_bound, min_lifespan = 99.9, 10 * 8760 # 8760 hours per year on average

    '''
    ===== 2) Model selection =====
    For our inference model we will use the classic electromigration model based on Black's equation. The current
    density is based on the custom IDFBCAMP chip designed by the Ivanov SoC lab.
    
    Physical parameters that are not directly related to wear-out are assumed to be already characterized
    (e.g., threshold voltage) and so are treated as fixed parameters, only the electromigration specific parameters form
    the latent variable space we infer.
    '''
    num_devices = 5

    # Express wire current density as a function of the number of transistors and the voltage applied
    def j_n(n_fins, vdd, vth_typ, i_base):
        return jnp.where(jnp.greater(vdd, vth_typ), n_fins * i_base * ((vdd - vth_typ) ** 2), 0.0)

    # The classic model for electromigration failure estimates, DOI: 10.1109/T-ED.1969.16754
    def em_blacks_equation(jn_wire, temp, em_n, em_eaa, wire_area, k):
        return (wire_area / (jn_wire ** em_n)) * jnp.exp((em_eaa * 0.01) / (k * temp))

    # Add inherent variability to electromigration sensor line failure times corresponding to 7% of the average time
    def em_variability(em_blacks, em_var_percent):
        return jnp.abs(em_var_percent * em_blacks)

    # Now correspond the degradation mechanisms to the output values
    mb.add_dependent('jn_wire', j_n)
    mb.add_dependent('em_blacks', em_blacks_equation)
    mb.add_dependent('em_variability', em_variability)
    mb.add_measured('em_ttf', dists.Normal, {'loc': 'em_blacks', 'scale': 'em_variability'}, num_devices)

    mb.add_hyperlatent('n_nom', dists.Normal, {'loc': 1.8, 'scale': 0.3})
    mb.add_hyperlatent('n_var', dists.Normal, {'loc': 0.1, 'scale': 0.02})
    mb.add_hyperlatent('eaa_nom', dists.Normal, {'loc': 2, 'scale': 0.3})
    mb.add_hyperlatent('eaa_var', dists.Normal, {'loc': 0.1, 'scale': 0.02})
    mb.add_latent('em_n', dists.Normal, {'loc': 'n_nom', 'scale': 'n_var'})
    mb.add_latent('em_eaa', dists.Normal, {'loc': 'eaa_nom', 'scale': 'eaa_var'})

    def fail_time(em_ttf):
        return jnp.min(em_ttf)

    mb.add_predictor('chip_fail', fail_time, dists.Normal, {'loc': 'chip_fail', 'scale': 'fail_var'},
                     pred_conds={'vdd': 0.85, 'temp': 330})

    # Wire area in nm^2
    mb.add_params(n_fins=24, vth_typ=0.32, i_base=0.8, wire_area=1.024 * 1000 * 1000, k=BOLTZ_EV,
                  em_var_percent=0.07, fail_var=12)
    model = mb.build_model()
    #tm = stratcona.TestDesignManager(mb, sample_per_dev=False)
    #tm_marg_sample = stratcona.TestDesignManager(mb, sample_per_dev=True)
    rng_key = rand.key(50)
    samples = model.sample_predictor_from_beliefs(rng_key, 'chip_fail', num_samples=300)
    print('Sampled!')
    # Can visualize the prior model and see how inference is required to achieve the required predictive confidence.
    if analyze_prior:
        #tm.set_experiment_conditions({'t1': {'vdd': 0.85, 'temp': 350}})
        #tm.examine('prior_predictive')
        #tm.override_func('life_sampler', tm._compiled_funcs['obs_sampler'])
        #tm.set_experiment_conditions({'t1': {'vdd': 0.8, 'temp': 330}})
        #estimate = tm.estimate_reliability(percent_bound, num_samples=30_000)
        #print(f"Estimated {percent_bound}% upper credible lifespan: {estimate} hours")
        if SHOW_PLOTS:
            plt.show()

    '''
    ===== 3) Resource limitation analysis =====
    To determine the bounds of the possible test design space, need to decide on what costs are acceptable. We'll say
    that the reliability team has been given an allocation of 5 chips to use for this qualification testing.
    
    TODO: May add time constraints later once I figure out BED and inference on censored data.
    
    The maximum and minimum voltages and temperatures that can be used are [0.7, 0.95] and [300, 400] respectively.
    '''
    temps = [300, 325, 350, 375, 400]
    volts = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    permute_conds = product(temps, volts)
    possible_tests = [{'t1': {'vdd': v, 'temp': t}} for t, v in permute_conds]
    exp_sampler = stratcona.assistants.iterator.iter_sampler(possible_tests)

    '''
    ===== 4) Accelerated test design analysis =====
    First the RIG must be determined, which depends on both the reliability target metric and the model being used. The
    latent variable space is normalized and adjusted to prioritize variables that impact the metric more, and the RIG is
    determined.
    
    Once we have BED statistics on all the possible tests we perform our risk analysis to determine which one to use.
    '''
    if run_bed_analysis:
        tm.curr_rig = 3.2

        # Compile the lifespan function
        field_vdd, field_temp = 0.8, 345
        #em_n, em_eaa = pt.dvector('em_n'), pt.dvector('em_eaa')
        em_n, em_eaa = pt.dscalar('em_n'), pt.dscalar('em_eaa')
        fail_time = em_blacks_equation(j_n(24, field_vdd), field_temp, em_n, em_eaa)
        #fail_time = pt.min(em_blacks_equation(j_n(24, field_vdd), field_temp, em_n, em_eaa))
        life_func = pytensor.function([em_n, em_eaa], fail_time)
        tm.override_func('life_func', life_func)
        tm.set_upper_credible_target(10 * 8760, 99.9)

        tm_marg_sample.compile_func('obs_sampler')
        #tm_marg_sample.examine('prior_predictive')
        #plt.show()
        #tm.override_func('obs_sampler', tm_marg_sample._compiled_funcs['obs_sampler'])
        # Run the experimental design analysis
        results = tm.determine_best_test(exp_sampler, num_tests_to_eval=15,
                                         num_obs_samples_per_test=3000, num_ltnt_samples_per_test=3000,
                                         life_target=min_lifespan)

        ## TODO: Improve score-based selection criteria
        def bed_score(pass_prob, fails_eig_gap, test_cost=0.0):
            return 1 / (((1 - pass_prob) * fails_eig_gap) + test_cost)
        results['final_score'] = bed_score(results['rig_pass_prob'], results['rig_fails_only_eig_gap'])
        selected_test = results.iloc[results['final_score'].idxmax()]['design']
    else:
        selected_test = {'t1': {'vdd': 0.90, 'temp': 375}}

    print(selected_test)

    '''
    ===== 5) Conduct the selected test =====
    Here we will use Gerabaldi to emulate running the test on a real-world product. The simulator only works with
    temporal models, thus our electromigration failures are determined in an entirely different manner than using a
    similar equation to the inference model.
    '''
    if simulated_data_mode == 'gerabaldi':
        em_meas = MeasSpec({'interconnect_failure': 5}, {}, 'Bed Height Sampling')
        selected_strs = StrsSpec({'vdd': selected_test['t1']['vdd'], 'temp': selected_test['t1']['temp']}, 1000, 'EM Stress')
        hundred_year_test = TestSpec([em_meas], 1, 1, name='Selected EM Test')
        hundred_year_test.append_steps([selected_strs, em_meas], 8760 * 100)

        test_env = PhysTestEnv(env_vrtns={'temp': EnvVrtnMdl(dev_vrtn_mdl=Normal(0, 0.6))})

        def em_voiding(time, vdd, temp, a, i_scale, t_scale):
            j = np.where(np.greater(vdd, vth_typ), 24 * i_base * ((vdd - vth_typ) ** 2), 0.0)
            voided_percent = a * (j ** i_scale) * ((temp - 200) ** t_scale) * 1e-7 * time
            return voided_percent

        def em_line_fail(init, cond, breaking_point, em_voiding):
            failed = np.where(np.greater(em_voiding, breaking_point), 1, init)
            return failed

        voiding_mdl = DegMechMdl(em_voiding,
                                 mdl_name='em_voiding',
                                 a=LatentVar(Normal(0.011, 0.0005)),
                                 i_scale=LatentVar(Normal(1, 0.05)),
                                 t_scale=LatentVar(Normal(1.1, 0.02)))
        em_mdl = DeviceMdl(DegPrmMdl(
            prm_name='interconnect_failure',
            init_val_mdl=InitValMdl(init_val=LatentVar(deter_val=0)),
            deg_mech_mdls=voiding_mdl,
            compute_eqn=em_line_fail,
            breaking_point=LatentVar(Normal(0.5, 0.002))
        ))

        emulated_test_data = gerabaldi.simulate(hundred_year_test, em_mdl, test_env)

        fails = []
        measd = emulated_test_data.measurements
        for dev in measd['device #'].unique():
            fail_step = measd.loc[measd['device #'] == dev]['measured'].ne(0).idxmax()
            if measd.loc[fail_step]['measured'] == 1:
                fails.append(measd.loc[fail_step]['time'].total_seconds() / 3600)
            else:
                fails.append(876_000)

    elif simulated_data_mode == 'model':
        # Generate the simulated data points using the same model type as the inference model to allow for validation,
        # since the target posterior is known in this case
        sim_vdd, sim_temp = selected_test['t1']['vdd'], selected_test['t1']['temp']
        em_n, em_eaa = pt.dvector('em_n'), pt.dvector('em_eaa')
        fail_time = em_blacks_equation(j_n(24, sim_vdd), sim_temp, em_n, em_eaa)
        life_func = pytensor.function([em_n, em_eaa], fail_time)

        rng = np.random.default_rng()
        n_sampled = rng.normal(1.65, 0.05, num_devices)
        eaa_sampled = rng.normal(2.05, 0.07, num_devices)
        base_fails = life_func(n_sampled, eaa_sampled)
        fails = rng.normal(base_fails, 0.05 * base_fails)
    else:
        # Use hard coded data
        fails = [120_000, 98_000, 456_000, 400_000, 234_000]

    print(f"Simulated failure times: {fails}")

    '''
    ===== 6) Perform model inference =====
    Update our model based on the emulated test data. First need to extract the failure times from the measurements of
    fail states.
    '''
    if run_inference:
        tm.set_experiment_conditions({'t1': {'vdd': selected_test['t1']['vdd'], 'temp': selected_test['t1']['temp']}})
        observed = {'t1': {'em_ttf': np.array(fails)}}
        print(tm.get_priors(for_user=True))
        tm.infer_model(observed)
        #tm.infer_model_custom_algo(observed)

    '''
    ===== 7) Prediction and confidence evaluation =====
    Check whether we meet the long-term reliability target and with sufficient confidence.
    '''
    if run_posterior_analysis:
        tm.compile_func('obs_sampler')
        tm.override_func('life_sampler', tm._compiled_funcs['obs_sampler'])
        tm.set_experiment_conditions({'t1': {'vdd': 0.8, 'temp': 330}})
        estimate = tm.estimate_reliability(percent_bound, num_samples=30_000)
        print(f"Estimated {percent_bound}% upper credible lifespan: {estimate} hours")
        if estimate > 10 * 8760:
            print('Yay! Computed metric meets spec.')
        else:
            print('Noooo! Computed metric fails to meet spec.')
        if SHOW_PLOTS:
            plt.show()

    '''
    ===== 8) Metrics reporting =====
    Generate the regulatory and consumer metrics we can use to market and/or certify the product.
    '''
    # TODO


if __name__ == '__main__':
    electromigration_qualification()
