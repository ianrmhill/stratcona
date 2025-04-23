# Copyright (c) 2025 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import time as t
from itertools import product
from functools import partial
import json

import numpyro
import numpyro.distributions as dists
# This call has to occur before importing jax
numpyro.set_host_device_count(4)

import jax.numpy as jnp # noqa: ImportNotAtTopOfFile

from matplotlib import pyplot as plt

import os
import sys
# This line adds the parent directory to the module search path so that the Stratcona module can be seen and imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import stratcona

BOLTZ_EV = 8.617e-5
CELSIUS_TO_KELVIN = 273.15
SHOW_PLOTS = False


def idfbcamp_qualification():
    analyze_prior = False
    run_bed_analysis = True
    run_inference = True
    run_posterior_analysis = True
    simulated_data_mode = 'model'

    '''
    ===== 1) Determine qualification objectives =====
    For this final case study we are not planning on selling the product, thus are more interested in simply gaining an
    understanding of degradation performance. For experimental utility, this means EIG more directly corresponds to
    utility, however it is important to consider balanced gain across the set of hyper-latent variables. We may want to
    ensure we learn a good bit about each variable rather than a lot about a few of them. 
    '''
    balanced_eig = 0
    # TODO: USE QX%-HDCR AND MINIMIZE THE REGION

    '''
    ===== 2) Physical model and prior belief elicitation =====
    '''
    mb = stratcona.SPMBuilder(mdl_name='IDFBCAMP Sensors')
    # TODO: Add priors!
    mb.add_params(k=BOLTZ_EV)
    # Model provided in JEDEC's JEP122H as general empirical NBTI degradation model, equation 5.3.1
    def nbti_vth(time, vdd, temp, nbti_a0, nbti_eaa, nbti_alpha, nbti_n, k, v_adj, t_adj):
        return (nbti_a0 * 0.001) * jnp.exp((nbti_eaa * 0.01) / (k * temp)) * ((vdd * v_adj) ** nbti_alpha) * ((time * t_adj) ** nbti_n)
    mb.add_hyperlatent('nbti_a0_nom', dists.Normal, {'loc': 4, 'scale': 1})
    mb.add_hyperlatent('nbti_a0_dev', dists.Normal, {'loc': 4, 'scale': 1})
    mb.add_hyperlatent('nbti_a0_chp', dists.Normal, {'loc': 4, 'scale': 1})
    mb.add_hyperlatent('nbti_eaa_nom', dists.Normal, {'loc': 4, 'scale': 1})
    mb.add_hyperlatent('nbti_eaa_dev', dists.Normal, {'loc': 4, 'scale': 1})
    mb.add_hyperlatent('nbti_eaa_chp', dists.Normal, {'loc': 4, 'scale': 1})
    mb.add_hyperlatent('nbti_alpha_nom', dists.Normal, {'loc': 4, 'scale': 1})
    mb.add_hyperlatent('nbti_alpha_dev', dists.Normal, {'loc': 4, 'scale': 1})
    mb.add_hyperlatent('nbti_alpha_chp', dists.Normal, {'loc': 4, 'scale': 1})
    mb.add_hyperlatent('nbti_n_nom', dists.Normal, {'loc': 4, 'scale': 1})
    mb.add_hyperlatent('nbti_n_dev', dists.Normal, {'loc': 4, 'scale': 1})
    mb.add_hyperlatent('nbti_n_chp', dists.Normal, {'loc': 4, 'scale': 1})
    mb.add_latent('nbti_a0', 'nbti_a0_nom', 'nbti_a0_dev', 'nbti_a0_chp')
    mb.add_latent('nbti_eaa', 'nbti_eaa_nom', 'nbti_eaa_dev', 'nbti_eaa_chp')
    mb.add_latent('nbti_alpha', 'nbti_alpha_nom', 'nbti_alpha_dev', 'nbti_alpha_chp')
    mb.add_latent('nbti_n', 'nbti_n_nom', 'nbti_n_dev', 'nbti_n_chp')

    def pbti_vth(time, vdd, temp, pbti_a0, pbti_eaa, pbti_alpha, pbti_n, k, v_adj, t_adj):
        return (pbti_a0 * 0.001) * jnp.exp((pbti_eaa * 0.01) / (k * temp)) * ((vdd * v_adj) ** pbti_alpha) * ((time * t_adj) ** pbti_n)
    mb.add_hyperlatent('pbti_a0_nom', dists.Normal, {'loc': 4, 'scale': 1})
    mb.add_hyperlatent('pbti_a0_dev', dists.Normal, {'loc': 4, 'scale': 1})
    mb.add_hyperlatent('pbti_a0_chp', dists.Normal, {'loc': 4, 'scale': 1})
    mb.add_hyperlatent('pbti_eaa_nom', dists.Normal, {'loc': 4, 'scale': 1})
    mb.add_hyperlatent('pbti_eaa_dev', dists.Normal, {'loc': 4, 'scale': 1})
    mb.add_hyperlatent('pbti_eaa_chp', dists.Normal, {'loc': 4, 'scale': 1})
    mb.add_hyperlatent('pbti_alpha_nom', dists.Normal, {'loc': 4, 'scale': 1})
    mb.add_hyperlatent('pbti_alpha_dev', dists.Normal, {'loc': 4, 'scale': 1})
    mb.add_hyperlatent('pbti_alpha_chp', dists.Normal, {'loc': 4, 'scale': 1})
    mb.add_hyperlatent('pbti_n_nom', dists.Normal, {'loc': 4, 'scale': 1})
    mb.add_hyperlatent('pbti_n_dev', dists.Normal, {'loc': 4, 'scale': 1})
    mb.add_hyperlatent('pbti_n_chp', dists.Normal, {'loc': 4, 'scale': 1})
    mb.add_latent('pbti_a0', 'pbti_a0_nom', 'pbti_a0_dev', 'pbti_a0_chp')
    mb.add_latent('pbti_eaa', 'pbti_eaa_nom', 'pbti_eaa_dev', 'pbti_eaa_chp')
    mb.add_latent('pbti_alpha', 'pbti_alpha_nom', 'pbti_alpha_dev', 'pbti_alpha_chp')
    mb.add_latent('pbti_n', 'pbti_n_nom', 'pbti_n_dev', 'pbti_n_chp')

    # HCI model from Takeda and Suzuki, DOI: https://doi.org/10.1109/EDL.1983.25667
    # Some basic negative temperature dependence term added to enrich the demo
    def phci_vth(time, vdd, temp, phci_h0, phci_alpha, phci_beta, phci_n, v_adj, t_adj):
        return (phci_h0 * 10) * jnp.exp(-phci_alpha / (vdd * v_adj)) * (temp ** -phci_beta) * ((time * t_adj) ** phci_n)
    mb.add_hyperlatent('phci_h0_nom', dists.Normal, {'loc': 4, 'scale': 1})
    mb.add_hyperlatent('phci_h0_dev', dists.Normal, {'loc': 4, 'scale': 1})
    mb.add_hyperlatent('phci_h0_chp', dists.Normal, {'loc': 4, 'scale': 1})
    mb.add_hyperlatent('phci_alpha_nom', dists.Normal, {'loc': 4, 'scale': 1})
    mb.add_hyperlatent('phci_alpha_dev', dists.Normal, {'loc': 4, 'scale': 1})
    mb.add_hyperlatent('phci_alpha_chp', dists.Normal, {'loc': 4, 'scale': 1})
    mb.add_hyperlatent('phci_beta_nom', dists.Normal, {'loc': 4, 'scale': 1})
    mb.add_hyperlatent('phci_beta_dev', dists.Normal, {'loc': 4, 'scale': 1})
    mb.add_hyperlatent('phci_beta_chp', dists.Normal, {'loc': 4, 'scale': 1})
    mb.add_hyperlatent('phci_n_nom', dists.Normal, {'loc': 4, 'scale': 1})
    mb.add_hyperlatent('phci_n_dev', dists.Normal, {'loc': 4, 'scale': 1})
    mb.add_hyperlatent('phci_n_chp', dists.Normal, {'loc': 4, 'scale': 1})
    mb.add_latent('phci_h0', 'phci_h0_nom', 'phci_h0_dev', 'phci_h0_chp')
    mb.add_latent('phci_alpha', 'phci_alpha_nom', 'phci_alpha_dev', 'phci_alpha_chp')
    mb.add_latent('phci_beta', 'phci_beta_nom', 'phci_beta_dev', 'phci_beta_chp')
    mb.add_latent('phci_n', 'phci_n_nom', 'phci_n_dev', 'phci_n_chp')

    def nhci_vth(time, vdd, temp, nhci_h0, nhci_alpha, nhci_beta, nhci_n, v_adj, t_adj):
        return (nhci_h0 * 10) * jnp.exp(-nhci_alpha / (vdd * v_adj)) * (temp ** -nhci_beta) * ((time * t_adj) ** nhci_n)
    mb.add_hyperlatent('nhci_h0_nom', dists.Normal, {'loc': 4, 'scale': 1})
    mb.add_hyperlatent('nhci_h0_dev', dists.Normal, {'loc': 4, 'scale': 1})
    mb.add_hyperlatent('nhci_h0_chp', dists.Normal, {'loc': 4, 'scale': 1})
    mb.add_hyperlatent('nhci_alpha_nom', dists.Normal, {'loc': 4, 'scale': 1})
    mb.add_hyperlatent('nhci_alpha_dev', dists.Normal, {'loc': 4, 'scale': 1})
    mb.add_hyperlatent('nhci_alpha_chp', dists.Normal, {'loc': 4, 'scale': 1})
    mb.add_hyperlatent('nhci_beta_nom', dists.Normal, {'loc': 4, 'scale': 1})
    mb.add_hyperlatent('nhci_beta_dev', dists.Normal, {'loc': 4, 'scale': 1})
    mb.add_hyperlatent('nhci_beta_chp', dists.Normal, {'loc': 4, 'scale': 1})
    mb.add_hyperlatent('nhci_n_nom', dists.Normal, {'loc': 4, 'scale': 1})
    mb.add_hyperlatent('nhci_n_dev', dists.Normal, {'loc': 4, 'scale': 1})
    mb.add_hyperlatent('nhci_n_chp', dists.Normal, {'loc': 4, 'scale': 1})
    mb.add_latent('nhci_h0', 'nhci_h0_nom', 'nhci_h0_dev', 'nhci_h0_chp')
    mb.add_latent('nhci_alpha', 'nhci_alpha_nom', 'nhci_alpha_dev', 'nhci_alpha_chp')
    mb.add_latent('nhci_beta', 'nhci_beta_nom', 'nhci_beta_dev', 'nhci_beta_chp')
    mb.add_latent('nhci_n', 'nhci_n_nom', 'nhci_n_dev', 'nhci_n_chp')

    # Sensor combined Vth shifts
    mb.add_intermediate('nbti_full_strs', partial(nbti_vth, v_adj=1.0, t_adj=1.0))
    mb.add_intermediate('nbti_hci_strs', partial(nbti_vth, v_adj=0.85, t_adj=1.0))
    mb.add_intermediate('pbti_full_strs', partial(pbti_vth, v_adj=1.0, t_adj=1.0))
    mb.add_intermediate('pbti_hci_strs', partial(pbti_vth, v_adj=0.85, t_adj=1.0))
    mb.add_intermediate('phci_full_strs', partial(phci_vth, v_adj=1.0, t_adj=1.0))
    mb.add_intermediate('nhci_full_strs', partial(nhci_vth, v_adj=1.0, t_adj=1.0))
    mb.add_params(vtn_typ=0.315, vtp_typ=0.325)
    def nbti_strs_vth(nbti_full_strs, vtp_typ):
        return vtp_typ - nbti_full_strs
    mb.add_intermediate('nbti_strs_vth', nbti_strs_vth)
    def pbti_strs_vth(pbti_full_strs, vtn_typ):
        return vtn_typ - pbti_full_strs
    mb.add_intermediate('pbti_strs_vth', pbti_strs_vth)
    def phci_strs_vth(nbti_hci_strs, phci_full_strs, vtp_typ):
        return vtp_typ - nbti_hci_strs - phci_full_strs
    mb.add_intermediate('phci_strs_vth', phci_strs_vth)
    def nhci_strs_vth(pbti_hci_strs, nhci_full_strs, vtn_typ):
        return vtn_typ - pbti_hci_strs - nhci_full_strs
    mb.add_intermediate('nhci_strs_vth', nhci_strs_vth)

    # Novel ring oscillator sensor readings
    mb.add_params(bti_ro_cl=0.6, hci_ro_cl=0.65, num_stages=51, k_avg=15.4, vdd_meas=0.8, ro_meas_stddev=1_000)
    # Following the RO inverter propagation delay analysis from MIT lecture: http://web.mit.edu/6.012/www/SP07-L13.pdf
    # The approximate per-stage delay is (1/2 * Q_L) / I_D for NMOS and PMOS with slightly different I_D, take their
    # average for typical delay. I_D is based on saturation mode as transistors haven't fully turned on yet, Q_L is
    # VDD * C_L (interconnect parasitics + next stage gate capacitance, in pF)
    def nbti_ro_freq(nbti_strs_vth, vdd_meas, bti_ro_cl, k_avg, vtp_typ, num_stages):
        nbti_d = (0.5 * vdd_meas * (bti_ro_cl * 1e-9)) / (k_avg * ((vdd_meas - (0.5 * (nbti_strs_vth + vtp_typ))) ** 2))
        return 1 / (0.5 * num_stages * nbti_d)
    mb.add_intermediate('nbti_ro_freq', nbti_ro_freq)
    mb.add_observed('nbti_ro_sensor', dists.Normal, {'loc': 'nbti_ro_freq', 'scale': 'ro_meas_stddev'}, 5)
    def pbti_ro_freq(pbti_strs_vth, vdd_meas, bti_ro_cl, k_avg, vtn_typ, num_stages):
        pbti_d = (0.5 * vdd_meas * (bti_ro_cl * 1e-9)) / (k_avg * ((vdd_meas - (0.5 * (pbti_strs_vth + vtn_typ))) ** 2))
        return 1 / (0.5 * num_stages * pbti_d)
    mb.add_intermediate('pbti_ro_freq', pbti_ro_freq)
    mb.add_observed('pbti_ro_sensor', dists.Normal, {'loc': 'pbti_ro_freq', 'scale': 'ro_meas_stddev'}, 5)
    def phci_ro_freq(phci_strs_vth, vdd_meas, hci_ro_cl, k_avg, vtp_typ, num_stages):
        phci_d = (0.5 * vdd_meas * (hci_ro_cl * 1e-9)) / (k_avg * ((vdd_meas - (0.5 * (phci_strs_vth + vtp_typ))) ** 2))
        return 1 / (0.5 * num_stages * phci_d)
    mb.add_intermediate('phci_ro_freq', phci_ro_freq)
    mb.add_observed('phci_ro_sensor', dists.Normal, {'loc': 'phci_ro_freq', 'scale': 'ro_meas_stddev'}, 5)
    def nhci_ro_freq(nhci_strs_vth, vdd_meas, hci_ro_cl, k_avg, vtn_typ, num_stages):
        nhci_d = (0.5 * vdd_meas * (hci_ro_cl * 1e-9)) / (k_avg * ((vdd_meas - (0.5 * (nhci_strs_vth + vtn_typ))) ** 2))
        return 1 / (0.5 * num_stages * nhci_d)
    mb.add_intermediate('nhci_ro_freq', nhci_ro_freq)
    mb.add_observed('nhci_ro_sensor', dists.Normal, {'loc': 'nhci_ro_freq', 'scale': 'ro_meas_stddev'}, 5)

    # Offset voltage sensor readings
    mb.add_params(namp_gain=5.4, pamp_gain=5.2, namp_vbase=0.614, pamp_vbase=0.410, gm=0.042, ro=400, adc_stddev=0.03)
    def nbti_vamp_out(nbti_strs_vth, namp_vbase, namp_gain, vtp_typ, gm, ro):
        return namp_vbase - (namp_gain * (-0.5 * (nbti_strs_vth - vtp_typ) * gm * (0.5 * ro)))
    mb.add_intermediate('nbti_vamp_out', nbti_vamp_out)
    mb.add_observed('nbti_voff_sensor', dists.Normal, {'loc': 'nbti_vamp_out', 'scale': 'adc_stddev'}, 5)
    def pbti_vamp_out(pbti_strs_vth, pamp_vbase, pamp_gain, vtn_typ, gm, ro):
        return pamp_vbase - (pamp_gain * (-0.5 * (pbti_strs_vth - vtn_typ) * gm * (0.5 * ro)))
    mb.add_intermediate('pbti_vamp_out', pbti_vamp_out)
    mb.add_observed('pbti_voff_sensor', dists.Normal, {'loc': 'pbti_vamp_out', 'scale': 'adc_stddev'}, 5)
    def phci_vamp_out(phci_strs_vth, namp_vbase, namp_gain, vtp_typ, gm, ro):
        return namp_vbase - (namp_gain * (-0.5 * (phci_strs_vth - vtp_typ) * gm * (0.5 * ro)))
    mb.add_intermediate('phci_vamp_out', phci_vamp_out)
    mb.add_observed('phci_voff_sensor', dists.Normal, {'loc': 'phci_vamp_out', 'scale': 'adc_stddev'}, 5)
    def nhci_vamp_out(nhci_strs_vth, pamp_vbase, pamp_gain, vtn_typ, gm, ro):
        return pamp_vbase - (pamp_gain * (-0.5 * (nhci_strs_vth - vtn_typ) * gm * (0.5 * ro)))
    mb.add_intermediate('nhci_vamp_out', nhci_vamp_out)
    mb.add_observed('nhci_voff_sensor', dists.Normal, {'loc': 'nhci_vamp_out', 'scale': 'adc_stddev'}, 5)

    # Package all the sensors together
    am = stratcona.AnalysisManager(mb.build_model(), rng_seed=49274575)
    am.set_field_use_conditions({'vdd': 0.8, 'temp': 330})

    # Visualize the predicted degradation curves subject to the high uncertainty of prior beliefs
    if analyze_prior:
        am.sim_test_meas_new(1000)

        if SHOW_PLOTS:
            plt.show()

    '''
    ===== 3) Resource limitation analysis =====
    Here the test bounds are nice and easy since they are real! I will be able to test 4 chips simultaneously in two
    batches of two, leading to fixed test dimensions. I then optimize over temperature, voltage, and time, bounded by
    available time of up to 730 hours (1 month), the room temp to 130C bounds of the test system, and 0.8V to 0.95V
    supported range of the chips.
    
    There are two sets of two chips, so with 5 temps, 4 volts, 4 times, there are 80 options per board, and thus 6400
    possible test configurations to consider.
    '''
    temps = [t + CELSIUS_TO_KELVIN for t in [30, 50, 70, 90, 110, 130]]
    volts = [0.8, 0.85, 0.9, 0.95]
    times = [130, 330, 530, 730]

    # TODO: Construct a reasonable set of possible test designs, probably not the full 6400
    permute_conds = product(temps, volts, times)
    possible_tests = [stratcona.TestDef(f'', {'t1': {'lot': 1, 'chp': 2}, 't2': {'lot': 1, 'chp': 2}},
        {'t1': {'vdd': v, 'temp': t, 'time': t}, 't2': {'vdd': v, 'temp': t, 'time': t}}) for t, v in permute_conds]
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
        results, perf_stats = am.determine_best_test_apr25(56, 101, 300, 100, exp_sampler, em_u_func)

        fig, ax = plt.subplots()
        eigs = jnp.array([res['utility']['eig'] for res in results]).reshape((len(temps), len(volts)))
        print(eigs)
        ax.contourf(volts, temps, eigs)
        ax.set_ylabel('Temperature')
        ax.set_xlabel('Voltage')
        #plt.show()

        simplified = {}
        for i, d in enumerate(results):
            simplified[str(i)] = {'Temp': float(d['design'].conds['t1']['temp']), 'Volt': float(d['design'].conds['t1']['vdd']),
                                  'EIG': float(d['utility']['eig']), 'MTRPP': float(d['utility']['qx_lbci_pp'])}
        with open('../bed_data/em_bed_evals.json', 'w') as f:
            json.dump(simplified, f)
        #def bed_score(pass_prob, fails_eig_gap, test_cost=0.0):
        #    return 1 / (((1 - pass_prob) * fails_eig_gap) + test_cost)
        #results['final_score'] = bed_score(results['rig_pass_prob'], results['rig_fails_only_eig_gap'])

        #selected_test = results.iloc[results['final_score'].idxmax()]['design']
        selected_test = results[2]
    else:
        selected_test = {'t1': {'vdd': 0.90, 'temp': 375}}

    am.set_test_definition(selected_test)

    '''
    ===== 5) Conduct the selected test =====
    
    '''
    real_data = False
    if real_data:
        sim_data = 0
    else:
        # Use hard coded data
        sim_data = [120_000, 98_000, 456_000, 400_000, 234_000]

    print(f"Simulated failure times: {sim_data}")

    '''
    ===== 6) Perform model inference =====
    Update our model based on the experimental data. First need to extract the failure times from the measurements of
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
    idfbcamp_qualification()
