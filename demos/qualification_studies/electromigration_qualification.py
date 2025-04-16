# Copyright (c) 2025 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import time as t
from itertools import product
from functools import partial

import numpyro
import numpyro.distributions as dists
# This call has to occur before importing jax
numpyro.set_host_device_count(4)

import jax.numpy as jnp # noqa: ImportNotAtTopOfFile
import jax.random as rand

import gerabaldi
from gerabaldi.models import *

from matplotlib import pyplot as plt

import os
import sys
# This line adds the parent directory to the module search path so that the Stratcona module can be seen and imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import stratcona

BOLTZ_EV = 8.617e-5
SHOW_PLOTS = False


def electromigration_qualification():
    analyze_prior = False
    run_bed_analysis = True
    run_inference = True
    run_posterior_analysis = True
    simulated_data_mode = 'model'

    '''
    ===== 1) Determine long-term reliability requirements =====
    We will base the requirement on the SIL3 specification of less than 10^-3 (0.001) probability of failure on demand,
    which in the context of long-term reliability we will interpret as a less than 0.1% probability that
    electromigration failures occur before the 10 year intended product useful life. This translates to a metric of a
    worst case 0.1% quantile credible region of at least 10 years.
    '''
    objective = stratcona.ReliabilityRequirement(metric=stratcona.engine.metrics.qx_lbci,
                                                 quantile=99.9, target_lifespan=10 * 8760)

    '''
    ===== 2) Model selection =====
    For our inference model we will use the classic electromigration model based on Black's equation. The current
    density is based on the custom IDFBCAMP chip designed by the Ivanov SoC lab.
    
    Physical parameters that are not directly related to wear-out are assumed to be already characterized
    (e.g., threshold voltage) and so are treated as fixed parameters, only the electromigration specific parameters form
    the latent variable space to learn.
    '''
    mb = stratcona.SPMBuilder(mdl_name='Black\'s Electromigration')
    num_devices = 5

    # Express wire current density as a function of the number of transistors and the voltage applied
    def j_n(n_fins, vdd, vth_typ, i_base):
        return jnp.where(jnp.greater(vdd, vth_typ), n_fins * i_base * ((vdd - vth_typ) ** 2), 0.0)

    # The classic model for electromigration failure estimates, DOI: 10.1109/T-ED.1969.16754
    def em_blacks_equation(jn_wire, temp, em_n, em_eaa, wire_area, k):
        return (wire_area / (jn_wire ** em_n)) * jnp.exp((em_eaa * 0.01) / (k * temp))

    # Now correspond the degradation mechanisms to the output values
    mb.add_intermediate('jn_wire', j_n)
    mb.add_intermediate('em_blacks', em_blacks_equation)
    mb.add_observed('em_ttf', dists.Normal, {'loc': 'em_blacks', 'scale': 'fail_var'}, num_devices)

    mb.add_hyperlatent('n_nom', dists.Normal, {'loc': 1.8, 'scale': 0.1})
    mb.add_hyperlatent('n_var', dists.Normal, {'loc': 0.1, 'scale': 0.05})
    mb.add_hyperlatent('eaa_nom', dists.Normal, {'loc': 8, 'scale': 0.2})
    mb.add_hyperlatent('eaa_var', dists.Normal, {'loc': 0.15, 'scale': 0.04})
    mb.add_latent('em_n', 'n_nom', 'n_var')
    mb.add_latent('em_eaa', 'eaa_nom', 'eaa_var')

    def fail_time(em_ttf):
        return jnp.min(em_ttf)

    mb.add_fail_criterion('lifespan', fail_time)

    # Wire area in nm^2
    mb.add_params(n_fins=48, vth_typ=0.32, i_base=0.8, wire_area=1.024 * 1000 * 1000, k=BOLTZ_EV, fail_var=16800)

    am = stratcona.AnalysisManager(mb.build_model(), rel_req=objective, rng_seed=2348923)
    am.set_field_use_conditions({'vdd': 0.85, 'temp': 330})

    # Can visualize the prior model and see how inference is required to achieve the required predictive confidence.
    if analyze_prior:
        am.evaluate_reliability('lifespan', plot_results=True)
        if SHOW_PLOTS:
            plt.show()

    '''
    ===== 3) Resource limitation analysis =====
    To determine the bounds of the possible test design space, need to decide on what costs are acceptable. We'll say
    that the reliability team has been given an allocation of 5 chips to use for this qualification testing.
    
    TODO: May add time constraints later once I figure out BED and inference on censored data.
    
    The maximum and minimum voltages and temperatures that can be used are [0.7, 0.95] and [300, 400] respectively.
    '''
    temps = [275, 300, 325, 350, 375, 400, 425]
    volts = [0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    permute_conds = product(temps, volts)
    possible_tests = [stratcona.TestDef('', {'t1': {'lot': 1, 'chp': 1}}, {'t1': {'vdd': v, 'temp': t}}) for t, v in permute_conds]
    exp_sampler = stratcona.assistants.iter_sampler(possible_tests)

    '''
    ===== 4) Accelerated test design analysis =====
    First the RIG must be determined, which depends on both the reliability target metric and the model being used. The
    latent variable space is normalized and adjusted to prioritize variables that impact the metric more, and the RIG is
    determined.
    
    Once we have BED statistics on all the possible tests we perform our risk analysis to determine which one to use.
    '''
    if run_bed_analysis:
        def em_u_func(ig, qx_lbci, trgt):
            eig = jnp.sum(ig) / ig.size
            mtrpp = jnp.sum(jnp.where(qx_lbci > trgt, 1, 0)) / qx_lbci.size
            return {'eig': eig, 'qx_lbci_pp': mtrpp}
        em_u_func = partial(em_u_func, trgt=am.relreq.target_lifespan)

        # Run the experimental design analysis
        results, perf_stats = am.determine_best_test_apr25(56, 201, 300, 200, exp_sampler, em_u_func)

        fig, ax = plt.subplots()
        eigs = jnp.array([res['utility']['eig'] for res in results]).reshape((len(temps), len(volts)))
        print(eigs)
        ax.contourf(volts, temps, eigs)
        ax.set_ylabel('Temperature')
        ax.set_xlabel('Voltage')
        plt.show()


        #def bed_score(pass_prob, fails_eig_gap, test_cost=0.0):
        #    return 1 / (((1 - pass_prob) * fails_eig_gap) + test_cost)
        #results['final_score'] = bed_score(results['rig_pass_prob'], results['rig_fails_only_eig_gap'])

        selected_test = results.iloc[results['final_score'].idxmax()]['design']
    else:
        selected_test = {'t1': {'vdd': 0.90, 'temp': 375}}

    am.set_test_definition(selected_test)
    print(am.test)

    '''
    ===== 5) Conduct the selected test =====
    Here we use Gerabaldi to emulate running the test on a real-world product. The simulator only works with
    temporal models, thus our electromigration failures are determined not by Black's equation but by an entirely
    different custom probabilistic model.
    
    Optional fallbacks to Black's equation or manual entry failure data are provided.
    '''
    if simulated_data_mode == 'gerabaldi':
        em_meas = MeasSpec({'interconnect_failure': 5}, {}, 'Bed Height Sampling')
        selected_strs = StrsSpec({'vdd': selected_test['t1']['vdd'], 'temp': selected_test['t1']['temp']}, 1000, 'EM Stress')
        hundred_year_test = TestSpec([em_meas], 1, 1, name='Selected EM Test')
        hundred_year_test.append_steps([selected_strs, em_meas], 8760 * 100)

        test_env = PhysTestEnv(env_vrtns={'temp': EnvVrtnMdl(dev_vrtn_mdl=Normal(0, 0.6))})

        def em_voiding(time, vdd, temp, a, i_scale, t_scale):
            j = jnp.where(jnp.greater(vdd, vth_typ), 24 * i_base * ((vdd - vth_typ) ** 2), 0.0)
            voided_percent = a * (j ** i_scale) * ((temp - 200) ** t_scale) * 1e-7 * time
            return voided_percent

        def em_line_fail(init, cond, breaking_point, em_voiding):
            failed = jnp.where(jnp.greater(em_voiding, breaking_point), 1, init)
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
        sim_priors = {'n_nom': {'loc': 1.65, 'scale': 0.002}, 'n_var': {'loc': 0.12, 'scale': 0.002},
                      'eaa_nom': {'loc': 8.5, 'scale': 0.002}, 'eaa_var': {'loc': 0.09, 'scale': 0.002}, }
        sim_data_full = am.sim_test_measurements(alt_priors=sim_priors)
        sim_data = {'e1': {'em_ttf': sim_data_full['e1_em_ttf']}, 'e2': {'em_ttf': sim_data_full['e2_em_ttf']}}
    else:
        # Use hard coded data
        sim_data = [120_000, 98_000, 456_000, 400_000, 234_000]

    print(f"Simulated failure times: {sim_data}")

    '''
    ===== 6) Perform model inference =====
    Update our model based on the emulated test data. First need to extract the failure times from the measurements of
    fail states.
    '''
    if run_inference:
        am.do_inference(sim_data)

    '''
    ===== 7) Prediction and confidence evaluation =====
    Check whether we meet the long-term reliability target and with sufficient confidence.
    '''
    if run_posterior_analysis:
        am.evaluate_reliability('chip_fail', plot_results=True)
        if SHOW_PLOTS:
            plt.show()

    '''
    ===== 8) Metrics reporting =====
    Generate the regulatory and consumer metrics we can use to market and/or certify the product.
    '''
    # TODO


if __name__ == '__main__':
    electromigration_qualification()
