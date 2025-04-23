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
    run_bed_analysis = False
    run_inference = True
    run_posterior_analysis = False
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
    mb.add_params(k=BOLTZ_EV, tempref=300)
    var_tf = dists.transforms.ComposeTransform([dists.transforms.SoftplusTransform(), dists.transforms.AffineTransform(0, 0.1)])
    # Model provided in JEDEC's JEP122H as general empirical NBTI degradation model, equation 5.3.1
    def nbti_vth(time, vdd, temp, nbti_a0, nbti_eaa, nbti_alpha, nbti_n, k, v_adj, t_adj):
        return (nbti_a0 * 0.01) * jnp.exp((nbti_eaa * -0.01) / (k * temp)) * ((vdd * v_adj) ** nbti_alpha) * ((time * t_adj) ** (nbti_n * 0.1))
    # Priors for a0_nom, a0_dev, a0_chp, eaa_nom, eaa_dev, alpha_nom, n_nom are all from IRPS inference case study posteriors
    # We exclude variations for some hyper-latents due to the data to model complexity discrepancy.
    mb.add_hyperlatent('nbti_a0_nom', dists.Normal, {'loc': 3.6, 'scale': 1.2})
    mb.add_hyperlatent('nbti_a0_dev', dists.Normal, {'loc': 5.3, 'scale': 3.5}, transform=var_tf)
    mb.add_hyperlatent('nbti_a0_chp', dists.Normal, {'loc': 10.6, 'scale': 3}, transform=var_tf)
    mb.add_hyperlatent('nbti_eaa_nom', dists.Normal, {'loc': 6.2, 'scale': 1.1})
    mb.add_hyperlatent('nbti_alpha_nom', dists.Normal, {'loc': 3.5, 'scale': 0.4})
    mb.add_hyperlatent('nbti_n_nom', dists.Normal, {'loc': 2, 'scale': 0.4})
    mb.add_latent('nbti_a0', 'nbti_a0_nom', 'nbti_a0_dev', 'nbti_a0_chp')
    mb.add_latent('nbti_eaa', 'nbti_eaa_nom')
    mb.add_latent('nbti_alpha', 'nbti_alpha_nom')
    mb.add_latent('nbti_n', 'nbti_n_nom')

    def pbti_vth(time, vdd, temp, pbti_a0, pbti_eaa, pbti_alpha, pbti_n, k, v_adj, t_adj):
        return (pbti_a0 * 0.01) * jnp.exp((pbti_eaa * -0.01) / (k * temp)) * ((vdd * v_adj) ** pbti_alpha) * ((time * t_adj) ** (pbti_n * 0.1))
    # Priors here are adapted from the NBTI priors but accounting for the observed trends reported in the sensor papers:
    # Lower HTOL degradation, higher voltage dependence, longer time constant
    mb.add_hyperlatent('pbti_a0_nom', dists.Normal, {'loc': 1.5, 'scale': 0.8})
    mb.add_hyperlatent('pbti_a0_dev', dists.Normal, {'loc': 3.3, 'scale': 2.1}, transform=var_tf)
    mb.add_hyperlatent('pbti_a0_chp', dists.Normal, {'loc': 5.6, 'scale': 1.3}, transform=var_tf)
    mb.add_hyperlatent('pbti_eaa_nom', dists.Normal, {'loc': 6.2, 'scale': 1.3})
    mb.add_hyperlatent('pbti_alpha_nom', dists.Normal, {'loc': 4.2, 'scale': 0.5})
    mb.add_hyperlatent('pbti_n_nom', dists.Normal, {'loc': 1.5, 'scale': 0.4})
    mb.add_latent('pbti_a0', 'pbti_a0_nom', 'pbti_a0_dev', 'pbti_a0_chp')
    mb.add_latent('pbti_eaa', 'pbti_eaa_nom')
    mb.add_latent('pbti_alpha', 'pbti_alpha_nom')
    mb.add_latent('pbti_n', 'pbti_n_nom')

    # HCI model from P. Zhang et al., DOI: https://doi.org/10.1109/TED.2017.2728083
    # Vds dependence removed for simplicity, merged into Vgs/Vdd term for our sensor designs
    def phci_vth(time, vdd, temp, phci_a0, phci_u, phci_alpha, phci_beta, tempref):
        return (phci_a0 * 0.001) * (vdd ** phci_u) * (time ** ((phci_alpha * 0.1) + ((phci_beta * 0.0001) * (temp - tempref))))
    # Priors taken from the P. Zhang et al. paper, adjusted slightly to try and match observed sensor behaviour
    mb.add_hyperlatent('phci_a0_nom', dists.Normal, {'loc': 15, 'scale': 7})
    mb.add_hyperlatent('phci_a0_dev', dists.Normal, {'loc': 5.3, 'scale': 2.5}, transform=var_tf)
    mb.add_hyperlatent('phci_a0_chp', dists.Normal, {'loc': 8.6, 'scale': 4}, transform=var_tf)
    mb.add_hyperlatent('phci_u_nom', dists.Normal, {'loc': 4.5, 'scale': 1})
    mb.add_hyperlatent('phci_alpha_nom', dists.Normal, {'loc': 1.2, 'scale': 0.3})
    mb.add_hyperlatent('phci_beta_nom', dists.Normal, {'loc': 6, 'scale': 2.5})
    mb.add_latent('phci_a0', 'phci_a0_nom', 'phci_a0_dev', 'phci_a0_chp')
    mb.add_latent('phci_u', 'phci_u_nom')
    mb.add_latent('phci_alpha', 'phci_alpha_nom')
    mb.add_latent('phci_beta', 'phci_beta_nom')

    def nhci_vth(time, vdd, temp, nhci_a0, nhci_u, nhci_alpha, nhci_beta, tempref):
        return (nhci_a0 * 0.001) * (vdd ** nhci_u) * (time ** ((nhci_alpha * 0.1) + ((nhci_beta * 0.0001) * (temp - tempref))))
    mb.add_hyperlatent('nhci_a0_nom', dists.Normal, {'loc': 11, 'scale': 7})
    mb.add_hyperlatent('nhci_a0_dev', dists.Normal, {'loc': 5.3, 'scale': 2.5}, transform=var_tf)
    mb.add_hyperlatent('nhci_a0_chp', dists.Normal, {'loc': 8.6, 'scale': 4}, transform=var_tf)
    mb.add_hyperlatent('nhci_u_nom', dists.Normal, {'loc': 4.5, 'scale': 1})
    mb.add_hyperlatent('nhci_alpha_nom', dists.Normal, {'loc': 1.2, 'scale': 0.3})
    mb.add_hyperlatent('nhci_beta_nom', dists.Normal, {'loc': 6, 'scale': 2.5})
    mb.add_latent('nhci_a0', 'nhci_a0_nom', 'nhci_a0_dev', 'nhci_a0_chp')
    mb.add_latent('nhci_u', 'nhci_u_nom')
    mb.add_latent('nhci_alpha', 'nhci_alpha_nom')
    mb.add_latent('nhci_beta', 'nhci_beta_nom')

    # Sensor combined Vth shifts
    mb.add_intermediate('nbti_full_strs', partial(nbti_vth, v_adj=1.0, t_adj=1.0))
    mb.add_intermediate('nbti_hci_strs', partial(nbti_vth, v_adj=0.85, t_adj=1.0))
    mb.add_intermediate('pbti_full_strs', partial(pbti_vth, v_adj=1.0, t_adj=1.0))
    mb.add_intermediate('pbti_hci_strs', partial(pbti_vth, v_adj=0.85, t_adj=1.0))
    mb.add_intermediate('phci_full_strs', partial(phci_vth))
    mb.add_intermediate('nhci_full_strs', partial(nhci_vth))
    mb.add_params(vtn_typ=0.315, vtp_typ=0.325)
    def nbti_strs_vth(nbti_full_strs, vtp_typ):
        return vtp_typ + nbti_full_strs
    mb.add_intermediate('nbti_strs_vth', nbti_strs_vth)
    def pbti_strs_vth(pbti_full_strs, vtn_typ):
        return vtn_typ + pbti_full_strs
    mb.add_intermediate('pbti_strs_vth', pbti_strs_vth)
    def phci_strs_vth(nbti_hci_strs, phci_full_strs, vtp_typ):
        return vtp_typ + nbti_hci_strs + phci_full_strs
    mb.add_intermediate('phci_strs_vth', phci_strs_vth)
    def nhci_strs_vth(pbti_hci_strs, nhci_full_strs, vtn_typ):
        return vtn_typ + pbti_hci_strs + nhci_full_strs
    mb.add_intermediate('nhci_strs_vth', nhci_strs_vth)

    # Novel ring oscillator sensor readings
    mb.add_params(bti_ro_cl=0.63, hci_ro_cl=0.65, num_stages=51, k_avg=13.6, vdd_meas=0.8, ro_meas_stddev=0.1)
    # Following the RO inverter propagation delay analysis from MIT lecture: http://web.mit.edu/6.012/www/SP07-L13.pdf
    # The approximate per-stage delay is (1/2 * Q_L) / I_D for NMOS and PMOS with slightly different I_D, take their
    # average for typical delay. I_D is based on saturation mode as transistors haven't fully turned on yet, Q_L is
    # VDD * C_L (interconnect parasitics + next stage gate capacitance, in pF)
    # Effective threshold voltage is the average of NMOS and PMOS, only one side stressed, thus 0.5 * (vtn + vtp + dvth)
    # k_avg captures the full 1/2 * W/L * mu * Cox
    def nbti_ro_freq(nbti_strs_vth, vdd_meas, bti_ro_cl, k_avg, vtn_typ, num_stages):
        nbti_d = (0.5 * vdd_meas * (bti_ro_cl * 1e-9)) / (k_avg * ((vdd_meas - (0.5 * (nbti_strs_vth + vtn_typ))) ** 2))
        return 0.000001 / (0.5 * num_stages * nbti_d)
    mb.add_intermediate('nbti_ro_freq', nbti_ro_freq)
    mb.add_observed('nbti_ro_sensor', dists.Normal, {'loc': 'nbti_ro_freq', 'scale': 'ro_meas_stddev'}, 5)
    def pbti_ro_freq(pbti_strs_vth, vdd_meas, bti_ro_cl, k_avg, vtp_typ, num_stages):
        pbti_d = (0.5 * vdd_meas * (bti_ro_cl * 1e-9)) / (k_avg * ((vdd_meas - (0.5 * (pbti_strs_vth + vtp_typ))) ** 2))
        return 0.000001 / (0.5 * num_stages * pbti_d)
    mb.add_intermediate('pbti_ro_freq', pbti_ro_freq)
    mb.add_observed('pbti_ro_sensor', dists.Normal, {'loc': 'pbti_ro_freq', 'scale': 'ro_meas_stddev'}, 5)
    def phci_ro_freq(phci_strs_vth, vdd_meas, hci_ro_cl, k_avg, vtn_typ, num_stages):
        phci_d = (0.5 * vdd_meas * (hci_ro_cl * 1e-9)) / (k_avg * ((vdd_meas - (0.5 * (phci_strs_vth + vtn_typ))) ** 2))
        return 0.000001 / (0.5 * num_stages * phci_d)
    mb.add_intermediate('phci_ro_freq', phci_ro_freq)
    mb.add_observed('phci_ro_sensor', dists.Normal, {'loc': 'phci_ro_freq', 'scale': 'ro_meas_stddev'}, 5)
    def nhci_ro_freq(nhci_strs_vth, vdd_meas, hci_ro_cl, k_avg, vtp_typ, num_stages):
        nhci_d = (0.5 * vdd_meas * (hci_ro_cl * 1e-9)) / (k_avg * ((vdd_meas - (0.5 * (nhci_strs_vth + vtp_typ))) ** 2))
        return 0.000001 / (0.5 * num_stages * nhci_d)
    mb.add_intermediate('nhci_ro_freq', nhci_ro_freq)
    mb.add_observed('nhci_ro_sensor', dists.Normal, {'loc': 'nhci_ro_freq', 'scale': 'ro_meas_stddev'}, 5)

    # Offset voltage sensor readings
    mb.add_params(namp_gain=5.4, pamp_gain=5.2, gm=0.022, ro=200, adc_stddev=0.03)
    def nbti_vamp_out(nbti_strs_vth, namp_gain, vtp_typ, gm, ro):
        return namp_gain * (0.5 * (nbti_strs_vth - vtp_typ) * gm * (0.5 * ro))
    mb.add_intermediate('nbti_vamp_out', nbti_vamp_out)
    mb.add_observed('nbti_voff_sensor', dists.Normal, {'loc': 'nbti_vamp_out', 'scale': 'adc_stddev'}, 5)
    def pbti_vamp_out(pbti_strs_vth, pamp_gain, vtn_typ, gm, ro):
        return pamp_gain * (0.5 * (pbti_strs_vth - vtn_typ) * gm * (0.5 * ro))
    mb.add_intermediate('pbti_vamp_out', pbti_vamp_out)
    mb.add_observed('pbti_voff_sensor', dists.Normal, {'loc': 'pbti_vamp_out', 'scale': 'adc_stddev'}, 5)
    def phci_vamp_out(phci_strs_vth, namp_gain, vtp_typ, gm, ro):
        return namp_gain * (0.5 * (phci_strs_vth - vtp_typ) * gm * (0.5 * ro))
    mb.add_intermediate('phci_vamp_out', phci_vamp_out)
    mb.add_observed('phci_voff_sensor', dists.Normal, {'loc': 'phci_vamp_out', 'scale': 'adc_stddev'}, 5)
    def nhci_vamp_out(nhci_strs_vth, pamp_gain, vtn_typ, gm, ro):
        return pamp_gain * (0.5 * (nhci_strs_vth - vtn_typ) * gm * (0.5 * ro))
    mb.add_intermediate('nhci_vamp_out', nhci_vamp_out)
    mb.add_observed('nhci_voff_sensor', dists.Normal, {'loc': 'nhci_vamp_out', 'scale': 'adc_stddev'}, 5)

    # Package all the sensors together
    am = stratcona.AnalysisManager(mb.build_model(), rng_seed=49374575)
    am.set_field_use_conditions({'vdd': 0.8, 'temp': 330})

    # Visualize the predicted degradation curves subject to the high uncertainty of prior beliefs
    if analyze_prior:
        simtest = stratcona.TestDef('priorsim', {'sim130': {'lot': 1, 'chp': 4}, 'sim330': {'lot': 1, 'chp': 4}, 'sim530': {'lot': 1, 'chp': 4}, 'sim730': {'lot': 1, 'chp': 4}},
                                    {'sim130': {'vdd': 1, 'temp': 400, 'time': 130}, 'sim330': {'vdd': 1, 'temp': 400, 'time': 330}, 'sim530': {'vdd': 1, 'temp': 400, 'time': 530}, 'sim730': {'vdd': 1, 'temp': 400, 'time': 730}})
        am.set_test_definition(simtest)
        simdata = am.sim_test_meas_new((100,))

        fig, ps = plt.subplots(1, 4)
        raw130 = ['sim130_nbti_voff_sensor', 'sim130_pbti_voff_sensor', 'sim130_phci_voff_sensor', 'sim130_nhci_voff_sensor']
        raw330 = ['sim330_nbti_voff_sensor', 'sim330_pbti_voff_sensor', 'sim330_phci_voff_sensor', 'sim330_nhci_voff_sensor']
        raw530 = ['sim530_nbti_voff_sensor', 'sim530_pbti_voff_sensor', 'sim530_phci_voff_sensor', 'sim530_nhci_voff_sensor']
        raw730 = ['sim730_nbti_voff_sensor', 'sim730_pbti_voff_sensor', 'sim730_phci_voff_sensor', 'sim730_nhci_voff_sensor']
        clrs = ['darkorange', 'sienna', 'gold', 'firebrick']
        for i in range(4):
            d1, d3, d5, d7, clr = simdata[raw130[i]].flatten(), simdata[raw330[i]].flatten(), simdata[raw530[i]].flatten(), simdata[raw730[i]].flatten(), clrs[i]
            ps[i].plot(jnp.full((len(d1),), 130), d1, color=clr, linestyle='', marker='.', markersize=5, label=None)
            ps[i].plot(jnp.full((len(d3),), 330), d3, color=clr, linestyle='', marker='.', markersize=5, label=None)
            ps[i].plot(jnp.full((len(d5),), 530), d5, color=clr, linestyle='', marker='.', markersize=5, label=None)
            ps[i].plot(jnp.full((len(d7),), 730), d7, color=clr, linestyle='', marker='.', markersize=5, label=None)
            ps[i].set_title(raw130[i])
        fig.tight_layout()
        plt.show()

    '''
    ===== 3) Resource limitation analysis =====
    Here the test bounds are nice and easy since they are real! I will be able to test 4 chips simultaneously in two
    batches of two, leading to fixed test dimensions. I then optimize over temperature, voltage, and time, bounded by
    available time of up to 730 hours (1 month), the room temp to 130C bounds of the test system, and 0.8V to 0.95V
    supported range of the chips.
    
    There are two sets of two chips, so with 5 temps, 5 volts, 4 times, there are 100 options per board, and thus 10_000
    possible test configurations to consider.
    '''
    temps = [t + CELSIUS_TO_KELVIN for t in [30, 50, 70, 90, 110, 130]]
    volts = [0.8, 0.85, 0.9, 0.95, 1.0]
    times = [130, 330, 530, 730]

    # TODO: Construct a reasonable set of possible test designs, probably not the full 6400
    #permute_conds = product(temps, volts, times)
    #possible_tests = [stratcona.TestDef(f'', {'t1': {'lot': 1, 'chp': 2}, 't2': {'lot': 1, 'chp': 2}},
    #    {'t1': {'vdd': v, 'temp': t, 'time': t}, 't2': {'vdd': v, 'temp': t, 'time': t}}) for t, v in permute_conds]
    #exp_sampler = stratcona.assistants.iter_sampler(possible_tests)

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

    inftest = stratcona.TestDef('sim', {'b1': {'lot': 1, 'chp': 2}, 'b2': {'lot': 1, 'chp': 2}},
                                {'b1': {'vdd': 1, 'temp': 403, 'time': 530}, 'b2': {'vdd': 0.9, 'temp': 363, 'time': 730}})
    am.set_test_definition(inftest)

    '''
    ===== 5) Conduct the selected test =====
    
    '''
    real_data = False
    if real_data:
        inf_data = 0
    else:
        simd = am.sim_test_meas_new()
        inf_data = {}
        inf_data['b1'] = {key[3:]: simd[key] for key in simd if 'b1_' in key}
        inf_data['b2'] = {key[3:]: simd[key] for key in simd if 'b2_' in key}
        print(inf_data)

    '''
    ===== 6) Perform model inference =====
    Update our model based on the experimental data. First need to extract the failure times from the measurements of
    fail states.
    '''
    if run_inference:
        am.do_inference_is(inf_data)
        print(am.relmdl.hyl_beliefs)

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
