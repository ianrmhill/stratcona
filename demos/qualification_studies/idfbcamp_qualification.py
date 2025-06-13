# Copyright (c) 2025 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import time as t
from itertools import product
from functools import partial
from multiprocess import Pool
import json

import numpyro
import numpyro.distributions as dists
# This call has to occur before importing jax
numpyro.set_host_device_count(4)

import jax
import jax.numpy as jnp # noqa: ImportNotAtTopOfFile
import jax.random as rand

import datetime as dt
import certifi
from pymongo import MongoClient
from pymongo.errors import PyMongoError

import seaborn as sb
from matplotlib import pyplot as plt
import pandas as pd

import os
import sys
# This line adds the parent directory to the module search path so that the Stratcona module can be seen and imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import stratcona

BOLTZ_EV = 8.617e-5
CELSIUS_TO_KELVIN = 273.15
SHOW_PLOTS = False
DB_NAME = 'stratcona'
COLL_NAME = 'idfbcamp-qual'


def login_to_database():
    tls_ca = certifi.where()
    uri = "mongodb+srv://arbutus.6v6mkhr.mongodb.net/?authSource=%24external&authMechanism=MONGODB-X509&retryWrites=true&w=majority"
    mongo_client = MongoClient(uri, tls=True, tlsCertificatekeyFile='../cert/mongo_cert.pem', tlsCAFile=tls_ca)
    db = mongo_client[DB_NAME]
    dataset = db[COLL_NAME]
    try:
        mongo_client.admin.command('ping')
    except PyMongoError as e:
        print(e)
        print("\nCould not connect to database successfully...")
    return dataset


def try_database_upload(dataset, formatted_data):
    try:
        dataset.insert_one(formatted_data)
    except PyMongoError as e:
        print(e)
        print(f"Encountered error trying to upload data to database at {dt.datetime.now(tz=dt.UTC)}")


def idfbcamp_qualification():
    analyze_prior = False
    run_bed_analysis = False
    examine_qual_data = True
    viz_exp_data = False
    run_inference = False
    run_posterior_analysis = False

    dataset = login_to_database()
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
    mbp = stratcona.SPMBuilder(mdl_name='IDFBCAMP PMOS Sensors')
    mbn = stratcona.SPMBuilder(mdl_name='IDFBCAMP NMOS Sensors')

    mbp.add_params(k=BOLTZ_EV, tempref=300)
    mbn.add_params(k=BOLTZ_EV, tempref=300)
    # Model provided in JEDEC's JEP122H as general empirical NBTI degradation model, equation 5.3.1
    def nbti_vth(time, vdd, temp, nbti_a0, nbti_eaa, nbti_alpha, nbti_n, k, v_adj, t_adj):
        return (nbti_a0 * 0.01) * jnp.exp((nbti_eaa * -0.01) / (k * temp)) * ((vdd * v_adj) ** nbti_alpha) * ((time * t_adj) ** (nbti_n * 0.1))
    # Priors for a0_nom, a0_dev, a0_chp, eaa_nom, eaa_dev, alpha_nom, n_nom are all from IRPS inference case study posteriors
    # We exclude variations for some hyper-latents due to the data to model complexity discrepancy.
    mbp.add_hyperlatent('nbti_a0_nom', dists.Normal, {'loc': 3.6, 'scale': 1.2})
    mbp.add_hyperlatent('nbti_eaa_nom', dists.Normal, {'loc': 6.2, 'scale': 0.3})
    mbp.add_hyperlatent('nbti_alpha_nom', dists.Normal, {'loc': 3.5, 'scale': 0.4})
    mbp.add_hyperlatent('nbti_n_nom', dists.Normal, {'loc': 2, 'scale': 0.4})
    mbp.add_latent('nbti_a0', 'nbti_a0_nom')
    mbp.add_latent('nbti_eaa', 'nbti_eaa_nom')
    mbp.add_latent('nbti_alpha', 'nbti_alpha_nom')
    mbp.add_latent('nbti_n', 'nbti_n_nom')

    def pbti_vth(time, vdd, temp, pbti_a0, pbti_eaa, pbti_alpha, pbti_n, k, v_adj, t_adj):
        return (pbti_a0 * 0.01) * jnp.exp((pbti_eaa * -0.01) / (k * temp)) * ((vdd * v_adj) ** pbti_alpha) * ((time * t_adj) ** (pbti_n * 0.1))
    # Priors here are adapted from the NBTI priors but accounting for the observed trends reported in the sensor papers:
    # Lower HTOL degradation, higher voltage dependence, longer time constant
    mbn.add_hyperlatent('pbti_a0_nom', dists.Normal, {'loc': 1.5, 'scale': 0.8})
    mbn.add_hyperlatent('pbti_eaa_nom', dists.Normal, {'loc': 6.2, 'scale': 0.3})
    mbn.add_hyperlatent('pbti_alpha_nom', dists.Normal, {'loc': 4.2, 'scale': 0.5})
    mbn.add_hyperlatent('pbti_n_nom', dists.Normal, {'loc': 1.5, 'scale': 0.4})
    mbn.add_latent('pbti_a0', 'pbti_a0_nom')
    mbn.add_latent('pbti_eaa', 'pbti_eaa_nom')
    mbn.add_latent('pbti_alpha', 'pbti_alpha_nom')
    mbn.add_latent('pbti_n', 'pbti_n_nom')

    # HCI model from P. Zhang et al., DOI: https://doi.org/10.1109/TED.2017.2728083
    # Vds dependence removed for simplicity, merged into Vgs/Vdd term for our sensor designs
    def phci_vth(time, vdd, temp, phci_a0, phci_u, phci_alpha, phci_beta, tempref):
        return (phci_a0 * 0.001) * (vdd ** phci_u) * (time ** ((phci_alpha * 0.1) + ((phci_beta * 0.0001) * (temp - tempref))))
    # Priors taken from the P. Zhang et al. paper, adjusted slightly to try and match observed sensor behaviour
    mbp.add_hyperlatent('phci_a0_nom', dists.Normal, {'loc': 15, 'scale': 7})
    mbp.add_hyperlatent('phci_u_nom', dists.Normal, {'loc': 4.5, 'scale': 1})
    mbp.add_hyperlatent('phci_alpha_nom', dists.Normal, {'loc': 1.2, 'scale': 0.3})
    mbp.add_hyperlatent('phci_beta_nom', dists.Normal, {'loc': 6, 'scale': 2.5})
    mbp.add_latent('phci_a0', 'phci_a0_nom')
    mbp.add_latent('phci_u', 'phci_u_nom')
    mbp.add_latent('phci_alpha', 'phci_alpha_nom')
    mbp.add_latent('phci_beta', 'phci_beta_nom')

    def nhci_vth(time, vdd, temp, nhci_a0, nhci_u, nhci_alpha, nhci_beta, tempref):
        return (nhci_a0 * 0.001) * (vdd ** nhci_u) * (time ** ((nhci_alpha * 0.1) + ((nhci_beta * 0.0001) * (temp - tempref))))
    mbn.add_hyperlatent('nhci_a0_nom', dists.Normal, {'loc': 11, 'scale': 7})
    mbn.add_hyperlatent('nhci_u_nom', dists.Normal, {'loc': 4.5, 'scale': 1})
    mbn.add_hyperlatent('nhci_alpha_nom', dists.Normal, {'loc': 1.2, 'scale': 0.3})
    mbn.add_hyperlatent('nhci_beta_nom', dists.Normal, {'loc': 6, 'scale': 2.5})
    mbn.add_latent('nhci_a0', 'nhci_a0_nom')
    mbn.add_latent('nhci_u', 'nhci_u_nom')
    mbn.add_latent('nhci_alpha', 'nhci_alpha_nom')
    mbn.add_latent('nhci_beta', 'nhci_beta_nom')

    # Sensor combined Vth shifts
    mbp.add_intermediate('nbti_full_strs', partial(nbti_vth, v_adj=1.0, t_adj=1.0))
    mbp.add_intermediate('nbti_hci_strs', partial(nbti_vth, v_adj=0.85, t_adj=1.0))
    mbp.add_intermediate('phci_full_strs', partial(phci_vth))
    mbp.add_params(vtn_typ=0.315, vtp_typ=0.325)
    mbn.add_intermediate('pbti_full_strs', partial(pbti_vth, v_adj=1.0, t_adj=1.0))
    mbn.add_intermediate('pbti_hci_strs', partial(pbti_vth, v_adj=0.85, t_adj=1.0))
    mbn.add_intermediate('nhci_full_strs', partial(nhci_vth))
    mbn.add_params(vtn_typ=0.315, vtp_typ=0.325)
    def nbti_strs_vth(nbti_full_strs, vtp_typ):
        return vtp_typ + nbti_full_strs
    mbp.add_intermediate('nbti_strs_vth', nbti_strs_vth)
    def pbti_strs_vth(pbti_full_strs, vtn_typ):
        return vtn_typ + pbti_full_strs
    mbn.add_intermediate('pbti_strs_vth', pbti_strs_vth)
    def phci_strs_vth(nbti_hci_strs, phci_full_strs, vtp_typ):
        return vtp_typ + nbti_hci_strs + phci_full_strs
    mbp.add_intermediate('phci_strs_vth', phci_strs_vth)
    def nhci_strs_vth(pbti_hci_strs, nhci_full_strs, vtn_typ):
        return vtn_typ + pbti_hci_strs + nhci_full_strs
    mbn.add_intermediate('nhci_strs_vth', nhci_strs_vth)

    # Novel ring oscillator sensor readings
    mbp.add_params(bti_ro_cl=0.63, hci_ro_cl=0.65, num_stages=51, k_avg=13.6, vdd_meas=0.8, ro_meas_stddev=2.0)
    mbn.add_params(bti_ro_cl=0.63, hci_ro_cl=0.65, num_stages=51, k_avg=13.6, vdd_meas=0.8, ro_meas_stddev=2.0)
    # Following the RO inverter propagation delay analysis from MIT lecture: http://web.mit.edu/6.012/www/SP07-L13.pdf
    # The approximate per-stage delay is (1/2 * Q_L) / I_D for NMOS and PMOS with slightly different I_D, take their
    # average for typical delay. I_D is based on saturation mode as transistors haven't fully turned on yet, Q_L is
    # VDD * C_L (interconnect parasitics + next stage gate capacitance, in pF)
    # Effective threshold voltage is the average of NMOS and PMOS, only one side stressed, thus 0.5 * (vtn + vtp + dvth)
    # k_avg captures the full 1/2 * W/L * mu * Cox
    def nbti_ro_freq(nbti_strs_vth, vdd_meas, bti_ro_cl, k_avg, vtn_typ, num_stages):
        nbti_d = (0.5 * vdd_meas * (bti_ro_cl * 1e-9)) / (k_avg * ((vdd_meas - (0.5 * (nbti_strs_vth + vtn_typ))) ** 2))
        return 0.000001 / (0.5 * num_stages * nbti_d)
    mbp.add_intermediate('nbti_ro_freq', nbti_ro_freq)
    mbp.add_observed('nbti_ro_sensor', dists.Normal, {'loc': 'nbti_ro_freq', 'scale': 'ro_meas_stddev'}, 5)
    def pbti_ro_freq(pbti_strs_vth, vdd_meas, bti_ro_cl, k_avg, vtp_typ, num_stages):
        pbti_d = (0.5 * vdd_meas * (bti_ro_cl * 1e-9)) / (k_avg * ((vdd_meas - (0.5 * (pbti_strs_vth + vtp_typ))) ** 2))
        return 0.000001 / (0.5 * num_stages * pbti_d)
    mbn.add_intermediate('pbti_ro_freq', pbti_ro_freq)
    mbn.add_observed('pbti_ro_sensor', dists.Normal, {'loc': 'pbti_ro_freq', 'scale': 'ro_meas_stddev'}, 5)
    def phci_ro_freq(phci_strs_vth, vdd_meas, hci_ro_cl, k_avg, vtn_typ, num_stages):
        phci_d = (0.5 * vdd_meas * (hci_ro_cl * 1e-9)) / (k_avg * ((vdd_meas - (0.5 * (phci_strs_vth + vtn_typ))) ** 2))
        return 0.000001 / (0.5 * num_stages * phci_d)
    mbp.add_intermediate('phci_ro_freq', phci_ro_freq)
    mbp.add_observed('phci_ro_sensor', dists.Normal, {'loc': 'phci_ro_freq', 'scale': 'ro_meas_stddev'}, 5)
    def nhci_ro_freq(nhci_strs_vth, vdd_meas, hci_ro_cl, k_avg, vtp_typ, num_stages):
        nhci_d = (0.5 * vdd_meas * (hci_ro_cl * 1e-9)) / (k_avg * ((vdd_meas - (0.5 * (nhci_strs_vth + vtp_typ))) ** 2))
        return 0.000001 / (0.5 * num_stages * nhci_d)
    mbn.add_intermediate('nhci_ro_freq', nhci_ro_freq)
    mbn.add_observed('nhci_ro_sensor', dists.Normal, {'loc': 'nhci_ro_freq', 'scale': 'ro_meas_stddev'}, 5)

    # Offset voltage sensor readings
    mbp.add_params(namp_gain=5.4, pamp_gain=5.2, gm=0.022, ro=200, adc_stddev=0.015)
    mbn.add_params(namp_gain=5.4, pamp_gain=5.2, gm=0.022, ro=200, adc_stddev=0.015)
    def nbti_vamp_out(nbti_strs_vth, namp_gain, vtp_typ, gm, ro):
        return namp_gain * (0.5 * (nbti_strs_vth - vtp_typ) * gm * (0.5 * ro))
    mbp.add_intermediate('nbti_vamp_out', nbti_vamp_out)
    mbp.add_observed('nbti_voff_sensor', dists.Normal, {'loc': 'nbti_vamp_out', 'scale': 'adc_stddev'}, 5)
    def pbti_vamp_out(pbti_strs_vth, pamp_gain, vtn_typ, gm, ro):
        return pamp_gain * (0.5 * (pbti_strs_vth - vtn_typ) * gm * (0.5 * ro))
    mbn.add_intermediate('pbti_vamp_out', pbti_vamp_out)
    mbn.add_observed('pbti_voff_sensor', dists.Normal, {'loc': 'pbti_vamp_out', 'scale': 'adc_stddev'}, 5)
    def phci_vamp_out(phci_strs_vth, namp_gain, vtp_typ, gm, ro):
        return namp_gain * (0.5 * (phci_strs_vth - vtp_typ) * gm * (0.5 * ro))
    mbp.add_intermediate('phci_vamp_out', phci_vamp_out)
    mbp.add_observed('phci_voff_sensor', dists.Normal, {'loc': 'phci_vamp_out', 'scale': 'adc_stddev'}, 5)
    def nhci_vamp_out(nhci_strs_vth, pamp_gain, vtn_typ, gm, ro):
        return pamp_gain * (0.5 * (nhci_strs_vth - vtn_typ) * gm * (0.5 * ro))
    mbn.add_intermediate('nhci_vamp_out', nhci_vamp_out)
    mbn.add_observed('nhci_voff_sensor', dists.Normal, {'loc': 'nhci_vamp_out', 'scale': 'adc_stddev'}, 5)

    # Generate the lifespan predictor variable
    def pmos_worst_deg(time, vdd, temp, vtp_typ,
                      nbti_a0, nbti_eaa, nbti_alpha, nbti_n, k,
                      phci_a0, phci_u, phci_alpha, phci_beta, tempref):

        nbti_partial = (nbti_a0 * 0.01) * jnp.exp((nbti_eaa * -0.01) / (k * temp)) * (time ** (nbti_n * 0.1))
        full_volt_coeff = ((vdd * 1.0) ** nbti_alpha)
        hci_volt_coeff = ((vdd * 0.85) ** nbti_alpha)
        nbti_full_vth = nbti_partial * full_volt_coeff
        nbti_hci_vth = nbti_partial * hci_volt_coeff
        phci_vth = (phci_a0 * 0.001) * (vdd ** phci_u) * (
                    time ** ((phci_alpha * 0.1) + ((phci_beta * 0.0001) * (temp - tempref))))

        nbti_strs_vth, phci_strs_vth = vtp_typ + nbti_full_vth, vtp_typ + nbti_hci_vth + phci_vth
        # The degradation threshold is hit by the worse of the two transistor stress configurations
        return jnp.maximum(nbti_strs_vth, phci_strs_vth)

    def deg_threshold(vdd, temp, vtp_typ, nbti_a0, nbti_eaa, nbti_alpha, nbti_n, k, phci_a0, phci_u, phci_alpha, phci_beta, tempref, threshold):
        #threshold = vtp_typ * 1.1
        func_args = {'vdd': vdd, 'temp': temp, 'vtp_typ': vtp_typ,
                      'nbti_a0': nbti_a0, 'nbti_eaa': nbti_eaa, 'nbti_alpha': nbti_alpha, 'nbti_n': nbti_n, 'k': k,
                      'phci_a0': phci_a0, 'phci_u': phci_u, 'phci_alpha': phci_alpha, 'phci_beta': phci_beta, 'tempref': tempref}
        def residue(time, **kwargs):
            return jnp.abs(threshold - pmos_worst_deg(time, **kwargs))
        t_life = stratcona.engine.minimization.minimize_jax(residue, func_args, (1, 1e7), precision=1e-4, log_gold=True)
        return t_life

    mbp.add_params(threshold=0.325 * 1.1)
    mbp.add_predictor('lifespan', deg_threshold)

    # Package all the sensors together
    amp = stratcona.AnalysisManager(mbp.build_model(), rng_seed=49374575)
    amp.set_field_use_conditions({'time': 10 * 8760, 'vdd': 0.8, 'temp': 330})
    amn = stratcona.AnalysisManager(mbn.build_model(), rng_seed=94372847)
    amn.set_field_use_conditions({'time': 10 * 8760, 'vdd': 0.8, 'temp': 330})

    # Analyze inverse
    check_inverse = False
    if check_inverse:
        k1, k2 = rand.split(rand.key(64728))
        sites = ('nbti_a0_nom', 'nbti_alpha_nom', 'nbti_eaa_nom', 'nbti_n_nom',
                 'phci_a0_nom', 'phci_alpha_nom', 'phci_beta_nom', 'phci_u_nom',
                 'field_nbti_voff_sensor_nbti_strs_vth', 'field_phci_voff_sensor_phci_strs_vth')
        sims = amp.relmdl.sample_new(k1, amp.field_test.dims, amp.field_test.conds, (), sites)
        amp.relmdl.param_vals['threshold'] = jnp.maximum(sims['field_nbti_voff_sensor_nbti_strs_vth'], sims['field_phci_voff_sensor_phci_strs_vth'])
        rv_sites = ('field_lifespan',)
        reverse = amp.relmdl.sample_new(k2, amp.field_test.dims, amp.field_test.conds, (), rv_sites, sims, compute_predictors=True)
        diff = reverse['field_lifespan'] - (10 * 8760)
        print(f'Forward-reverse mismatch: {diff} hours')

    # Visualize the predicted degradation curves subject to the high uncertainty of prior beliefs
    if analyze_prior:
        simtest = stratcona.TestDef('priorsim', {'sim130': {'lot': 1, 'chp': 4}, 'sim330': {'lot': 1, 'chp': 4}, 'sim530': {'lot': 1, 'chp': 4}, 'sim730': {'lot': 1, 'chp': 4}},
                                    {'sim130': {'vdd': 1, 'temp': 400, 'time': 130}, 'sim330': {'vdd': 1, 'temp': 400, 'time': 330}, 'sim530': {'vdd': 1, 'temp': 400, 'time': 530}, 'sim730': {'vdd': 1, 'temp': 400, 'time': 730}})
        amp.set_test_definition(simtest)
        amn.set_test_definition(simtest)
        pdata = amp.sim_test_meas_new((100,))
        ndata = amn.sim_test_meas_new((100,))

        fig, ps = plt.subplots(1, 4)
        raw130 = ['sim130_nbti_ro_sensor', 'sim130_pbti_ro_sensor', 'sim130_phci_ro_sensor', 'sim130_nhci_ro_sensor']
        raw330 = ['sim330_nbti_ro_sensor', 'sim330_pbti_ro_sensor', 'sim330_phci_ro_sensor', 'sim330_nhci_ro_sensor']
        raw530 = ['sim530_nbti_ro_sensor', 'sim530_pbti_ro_sensor', 'sim530_phci_ro_sensor', 'sim530_nhci_ro_sensor']
        raw730 = ['sim730_nbti_ro_sensor', 'sim730_pbti_ro_sensor', 'sim730_phci_ro_sensor', 'sim730_nhci_ro_sensor']
        clrs = ['darkorange', 'sienna', 'gold', 'firebrick']
        for i in range(4):
            simdata = pdata if i % 2 == 0 else ndata
            d1, d3, d5, d7, clr = simdata[raw130[i]].flatten(), simdata[raw330[i]].flatten(), simdata[raw530[i]].flatten(), simdata[raw730[i]].flatten(), clrs[i]
            ps[i].plot(jnp.full((len(d1),), 130), d1, color=clr, linestyle='', marker='.', markersize=5, label=None)
            ps[i].plot(jnp.full((len(d3),), 330), d3, color=clr, linestyle='', marker='.', markersize=5, label=None)
            ps[i].plot(jnp.full((len(d5),), 530), d5, color=clr, linestyle='', marker='.', markersize=5, label=None)
            ps[i].plot(jnp.full((len(d7),), 730), d7, color=clr, linestyle='', marker='.', markersize=5, label=None)
            ps[i].set_title(raw130[i])
        fig.tight_layout()
        plt.show()

    # Try generating some predictive lifespans
    kx, kz = rand.split(rand.key(7342))
    n_x, n_z = 4, 5
    fd_dims, fd_conds = amp.field_test.dims, amp.field_test.conds
    x_s = amp.relmdl.sample_new(kx, fd_dims, fd_conds, batch_dims=(n_x, n_z), keep_sites=amp.relmdl.hyls)
    z_s = amp.relmdl.sample_new(kz, fd_dims, fd_conds, (n_x, n_z), keep_sites=amp.relmdl.predictors,
                                conditionals=x_s, compute_predictors=True)
    print(f'Generated lifespans: {z_s}\n')

    '''
    ===== 3) Resource limitation analysis =====
    Here the test bounds are nice and easy since they are real! I will be able to test 4 chips simultaneously in two
    batches of two, leading to fixed test dimensions. I then optimize over temperature, voltage, and time, bounded by
    available time of up to 730 hours (1 month), the room temp to 130C bounds of the test system, and 0.8V to 0.95V
    supported range of the chips.
    
    There are two sets of two chips, so with 5 temps, 5 volts, 4 times, there are 100 options per board, and thus 10_000
    possible test configurations to consider.
    '''
    temps = [t + CELSIUS_TO_KELVIN for t in [30, 55, 80, 105, 130]]
    volts = [0.8, 0.85, 0.9, 0.95, 1.0]
    times = [130, 330, 530, 730]

    permute_conds = product(temps, temps, volts, volts, times, times)
    test_conds_list = [{'b1': {'temp': c1, 'vdd': v1, 'time': t1}, 'b2': {'temp': c2, 'vdd': v2, 'time': t2}} for c1, c2, v1, v2, t1, t2 in permute_conds]
    possible_tests = [stratcona.TestDef('', {'b1': {'lot': 1, 'chp': 2}, 'b2': {'lot': 1, 'chp': 2}}, conds) for conds in test_conds_list]

    '''
    ===== 4) Accelerated test design analysis =====
    First the RIG must be determined, which depends on both the reliability target metric and the model being used. The
    latent variable space is normalized and adjusted to prioritize variables that impact the metric more, and the RIG is
    determined.
    
    Once we have BED statistics on all the possible tests we perform our risk analysis to determine which one to use.
    '''
    if run_bed_analysis:
        def p_u_func(ig, qx_hdcr_width):
            eig = jnp.sum(ig) / ig.size
            vig = jnp.sum(((ig - eig) ** 2) / ig.size)
            mig = jnp.min(ig)
            pred_width = jnp.sum(qx_hdcr_width) / qx_hdcr_width.size
            return {'eig': eig, 'vig': vig, 'mig': mig, 'e_qx_hdcr_width': pred_width}
        def n_u_func(ig):
            eig = jnp.sum(ig) / ig.size
            vig = jnp.sum(((ig - eig) ** 2) / ig.size)
            mig = jnp.min(ig)
            return {'eig': eig, 'vig': vig, 'mig': mig}

        batches = 5
        d_batch_size = 10
        exp_samplers_p = [stratcona.assistants.iter_sampler(possible_tests[i*d_batch_size:(i*d_batch_size)+d_batch_size]) for i in range(batches)]
        exp_samplers_n = [stratcona.assistants.iter_sampler(possible_tests[i*d_batch_size:(i*d_batch_size)+d_batch_size]) for i in range(batches)]
        keys = rand.split(amp._derive_key(), batches)
        eval_d_batch_p = partial(stratcona.engine.bed.pred_bed_apr25, n_d=d_batch_size, n_y=1_000, n_v=1, n_x=10_000, spm=amp.relmdl,
                                 utility=p_u_func, field_d=amp.field_test)
        eval_d_batch_n = partial(stratcona.engine.bed.pred_bed_apr25, n_d=d_batch_size, n_y=1_000, n_v=1, n_x=10_000, spm=amn.relmdl,
                                 utility=n_u_func, field_d=amn.field_test)

        # Run the experimental design analysis in batched segments to allow for checkpointing
        for i in range(batches):
            res_p, perf_p = eval_d_batch_p(keys[i], exp_samplers_p[i])
            res_n, perf_n = eval_d_batch_n(keys[i], exp_samplers_n[i])
            simplified = {'batch': i, 'batch-size': d_batch_size, 'submit-time': str(dt.datetime.now(tz=dt.UTC)),
                          'n-y': 1_000, 'n-x': 10_000}
            for j in range(d_batch_size):
                simplified[f'{str(i)}-{str(j)}'] = {
                    'c1': float(res_p[j]['design'].conds['b1']['temp']), 'c2': float(res_p[j]['design'].conds['b2']['temp']),
                    'v1': float(res_p[j]['design'].conds['b1']['vdd']), 'v2': float(res_p[j]['design'].conds['b2']['vdd']),
                    't1': float(res_p[j]['design'].conds['b1']['time']), 't2': float(res_p[j]['design'].conds['b2']['time']),
                    'eig-p': float(res_p[j]['utility']['eig']), 'eig-n': float(res_n[j]['utility']['eig']),
                    'vig-p': float(res_p[j]['utility']['vig']), 'vig-n': float(res_n[j]['utility']['vig']),
                    'mig-p': float(res_p[j]['utility']['mig']), 'mig-n': float(res_n[j]['utility']['mig']),
                    'e-hdcr-width-p': float(res_p[j]['utility']['e_qx_hdcr_width'])}
            with open(f'../bed_data/idfbcamp_bed_evals_batch{i}.json', 'w') as f:
                json.dump(simplified, f)
            try_database_upload(dataset, simplified)

    if examine_qual_data:
        ds = []
        for i in range(50):
            with open(f'../ccdata/bed_qual_y5k_x50k_batch{i}.json', 'r') as f:
                batch = json.load(f)
            for d in [k for k in batch.keys() if k[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']]:
                ds.append(batch[d])

        # Normalize the mean and variance of HDCR width to match EIG
        widths = jnp.array([d['e-hdcr-width-p'] for d in ds])
        eigs = jnp.array([d['eig-p'] + d['eig-n'] for d in ds])
        w_m, w_v = jnp.mean(widths), jnp.std(widths)
        eig_m, eig_v = jnp.mean(eigs), jnp.std(eigs)

        for d in ds:
            d['w-score'] = (((w_m - d['e-hdcr-width-p']) / w_v) * eig_v) + eig_m
            d['u'] = (d['eig-p'] + d['eig-n']) + d['w-score']

        u_max, eig_max, w_max = 0, 0, 0
        i_u_max, i_eig_max, i_w_max = -1, -1, -1
        for i, d in enumerate(ds):
            if d['u'] > u_max:
                u_max = d['u']
                i_u_max = i
            if d['eig-p'] + d['eig-n'] > eig_max:
                eig_max = d['eig-p'] + d['eig-n']
                i_eig_max = i
            if d['w-score'] > w_max:
                w_max = d['w-score']
                i_w_max = i
        print(ds[i_u_max])
        high_us = [d for d in ds if d['u'] > u_max * 0.99]

        slice_time, slice_vdd, slice_temp = 730, 1.0, 403.15
        slice = [d for d in ds if d['t1'] == slice_time and d['t2'] == slice_time
                 and d['c1'] == slice_temp and d['c2'] == slice_temp]
        # and e['v1'] == slice_vdd and e['v2'] == slice_vdd]
        t1 = t2 = [303.15, 328.15, 353.15, 378.15, 403.15]
        v1 = v2 = [0.8, 0.85, 0.9, 0.95, 1.0]
        u = jnp.array([d['u'] for d in slice]).reshape((5, 5))

        fig, ax = plt.subplots()
        ax.contourf(v1, v2, u)
        ax.set_ylabel('v1')
        ax.set_xlabel('v2')
        plt.show()

    '''
    ===== 5) Conduct the selected test =====
    '''
    # This file is the real data!!!
    with open('../bed_data/idfbcamp_qual_for_inf.json', 'r') as f:
        obs_data = json.load(f)

    obstest = stratcona.TestDef('dopt', {'b1': {'lot': 1, 'chp': 2}, 'b2': {'lot': 1, 'chp': 2}},
                                {'b1': {'vdd': 1.0, 'temp': 130 + CELSIUS_TO_KELVIN, 'time': 730},
                                'b2': {'vdd': 0.8, 'temp': 130 + CELSIUS_TO_KELVIN, 'time': 730}})
    amp.set_test_definition(obstest)
    amn.set_test_definition(obstest)

    bti_init, hci_init = 487.61908, 472.61542
    for b in obs_data:
        for obs in obs_data[b]:
            if obs in ['nbti_ro_sensor', 'pbti_ro_sensor']:
                obs_data[b][obs] = jnp.array(obs_data[b][obs])
                obs_data[b][obs] = bti_init + (obs_data[b][obs] * bti_init)
            elif obs in ['phci_ro_sensor', 'nhci_ro_sensor']:
                obs_data[b][obs] = jnp.array(obs_data[b][obs])
                obs_data[b][obs] = hci_init + (obs_data[b][obs] * hci_init)
            elif obs in ['nbti_voff_sensor', 'phci_voff_sensor']:
                obs_data[b][obs] = jnp.array(obs_data[b][obs])
                obs_data[b][obs] = obs_data[b][obs] * -5
            elif obs in ['pbti_voff_sensor', 'nhci_voff_sensor']:
                obs_data[b][obs] = jnp.array(obs_data[b][obs])
                obs_data[b][obs] = obs_data[b][obs] * -5

    if viz_exp_data:
        resn = amn.sim_test_meas_new()
        resp = amp.sim_test_meas_new()

        df1 = pd.DataFrame([(k, float(v)) for k, arr in obs_data['b1'].items() for v in arr.flatten()],
                          columns=["param", "measured"])
        df2 = pd.DataFrame([(k, float(v)) for k, arr in obs_data['b2'].items() for v in arr.flatten()],
                          columns=["param", "measured"])
        resa = resn | resp
        dfs = pd.DataFrame([(k, float(v)) for k, arr in resa.items() for v in arr.flatten()],
                          columns=["param", "measured"])

        colors = ['cornflowerblue', 'mediumorchid', 'turquoise', 'hotpink',
                   'mediumblue', 'darkviolet', 'lightseagreen', 'deeppink']
        fig, p = plt.subplots(4, 1)

        df_vth_b1 = df1.loc[df1['param'].isin(['pbti_voff_sensor', 'nhci_voff_sensor', 'nbti_voff_sensor', 'phci_voff_sensor'])]
        df_vth_b1 = pd.concat((df_vth_b1, dfs.loc[dfs['param'].isin(['b1_pbti_voff_sensor', 'b1_nhci_voff_sensor', 'b1_nbti_voff_sensor', 'b1_phci_voff_sensor'])]))
        sb.stripplot(df_vth_b1, x='param', y='measured', hue='param', ax=p[0], alpha=0.8, palette=colors)

        df_vth_b2 = df2.loc[df2['param'].isin(['pbti_voff_sensor', 'nhci_voff_sensor', 'nbti_voff_sensor', 'phci_voff_sensor'])]
        df_vth_b2 = pd.concat((df_vth_b2, dfs.loc[dfs['param'].isin(['b2_pbti_voff_sensor', 'b2_nhci_voff_sensor', 'b2_nbti_voff_sensor', 'b2_phci_voff_sensor'])]))
        sb.stripplot(df_vth_b2, x='param', y='measured', hue='param', ax=p[1], alpha=0.8, palette=colors)

        df_ro_b1 = df1.loc[df1['param'].isin(['pbti_ro_sensor', 'nhci_ro_sensor', 'nbti_ro_sensor', 'phci_ro_sensor'])]
        df_ro_b1 = pd.concat((df_ro_b1, dfs.loc[dfs['param'].isin(['b1_pbti_ro_sensor', 'b1_nhci_ro_sensor', 'b1_nbti_ro_sensor', 'b1_phci_ro_sensor'])]))
        sb.stripplot(df_ro_b1, x='param', y='measured', hue='param', ax=p[2], alpha=0.8, palette=colors)

        df_ro_b2 = df2.loc[df2['param'].isin(['pbti_ro_sensor', 'nhci_ro_sensor', 'nbti_ro_sensor', 'phci_ro_sensor'])]
        df_ro_b2 = pd.concat((df_ro_b2, dfs.loc[dfs['param'].isin(['b2_pbti_ro_sensor', 'b2_nhci_ro_sensor', 'b2_nbti_ro_sensor', 'b2_phci_ro_sensor'])]))
        sb.stripplot(df_ro_b2, x='param', y='measured', hue='param', ax=p[3], alpha=0.8, palette=colors)

        plt.show()

    '''
    ===== 6) Perform model inference =====
    Update our model based on the experimental data. First need to extract the failure times from the measurements of
    fail states.
    '''
    pri_p = amp.relmdl.hyl_beliefs
    pri_n = amn.relmdl.hyl_beliefs
    if run_inference:
        amp.do_inference(obs_data)
        print(f'Old p: {pri_p}')
        print(f'New p: {amp.relmdl.hyl_beliefs}')
        pst_p = amp.relmdl.hyl_beliefs

        with open('../bed_data/idfbcamp_pst_hyls_p.json', 'w') as f:
            flt_p = {}
            for hyl in pst_p:
                flt_p[hyl] = {prm: float(pst_p[hyl][prm]) for prm in pst_p[hyl]}
            json.dump(flt_p, f)

        amn.do_inference(obs_data)
        print(f'Old n: {pri_n}')
        print(f'New n: {amn.relmdl.hyl_beliefs}')
        pst_n = amn.relmdl.hyl_beliefs

        with open('../bed_data/idfbcamp_pst_hyls_n.json', 'w') as f:
            flt_n = {}
            for hyl in pst_n:
                flt_n[hyl] = {prm: float(pst_n[hyl][prm]) for prm in pst_n[hyl]}
            json.dump(flt_n, f)

        jax.clear_caches()

    '''
    ===== 7) Prediction and confidence evaluation =====
    Check whether we meet the long-term reliability target and with sufficient confidence.
    '''
    if run_posterior_analysis:
        if True:
            with open('../bed_data/idfbcamp_pst_hyls_p.json', 'r') as f:
                pst_p = json.load(f)
            with open('../bed_data/idfbcamp_pst_hyls_n.json', 'r') as f:
                pst_n = json.load(f)

        amn.relmdl.hyl_beliefs = pst_n
        amp.relmdl.hyl_beliefs = pst_p
        jax.clear_caches()

        amp.relreq = stratcona.ReliabilityRequirement(stratcona.engine.metrics.qx_hdcr_l, 90, 87_600)
        n_hyl = 1_000_000

        def lp_fn(vals, site, key, test):
            return amn.relmdl.logp_new(rng_key=key, test_dims=test.dims, test_conds=test.conds, site_vals={site: vals},
                                       conditional=None, batch_dims=(len(vals),))
        def lp_fp(vals, site, key, test):
            return amp.relmdl.logp_new(rng_key=key, test_dims=test.dims, test_conds=test.conds, site_vals={site: vals},
                                       conditional=None, batch_dims=(len(vals),))

        k1, k2 = rand.split(amn._derive_key(), 2)
        n_hyls = ('pbti_a0_nom', 'pbti_eaa_nom', 'pbti_alpha_nom', 'pbti_n_nom',
                  'nhci_a0_nom', 'nhci_u_nom', 'nhci_alpha_nom', 'nhci_beta_nom')
        p_hyls = ('nbti_a0_nom', 'nbti_eaa_nom', 'nbti_alpha_nom', 'nbti_n_nom',
                  'phci_a0_nom', 'phci_u_nom', 'phci_alpha_nom', 'phci_beta_nom')
        hyln = amn.relmdl.sample_new(k1, amn.test.dims, amn.test.conds, (n_hyl,), n_hyls)
        hylp = amp.relmdl.sample_new(k2, amp.test.dims, amp.test.conds, (n_hyl,), p_hyls)

        pst_entropy, pri_entropy = {}, {}
        for hyl in n_hyls:
            pst_entropy[hyl] = stratcona.engine.bed.entropy(
                hyln[hyl], partial(lp_fn, site=hyl, test=amn.test, key=k1),
                limiting_density_range=(-838.8608, 838.8607))
        for hyl in p_hyls:
            pst_entropy[hyl] = stratcona.engine.bed.entropy(
                hylp[hyl], partial(lp_fp, site=hyl, test=amp.test, key=k1),
                limiting_density_range=(-838.8608, 838.8607))

        pst_h_tot = sum(pst_entropy.values())
        pst_life = amp.evaluate_reliability('lifespan', num_samples=1_000_000)

        # Now compute prior entropy and lifespan distribution for comparison
        amn.relmdl.hyl_beliefs = pri_n
        amp.relmdl.hyl_beliefs = pri_p
        jax.clear_caches()

        k1, k2 = rand.split(amn._derive_key(), 2)
        hyln = amn.relmdl.sample_new(k1, amn.test.dims, amn.test.conds, (n_hyl,), n_hyls)
        hylp = amp.relmdl.sample_new(k2, amp.test.dims, amp.test.conds, (n_hyl,), p_hyls)

        for hyl in n_hyls:
            pri_entropy[hyl] = stratcona.engine.bed.entropy(
                hyln[hyl], partial(lp_fn, site=hyl, test=amn.test, key=k1),
                limiting_density_range=(-838.8608, 838.8607))
        for hyl in p_hyls:
            pri_entropy[hyl] = stratcona.engine.bed.entropy(
                hylp[hyl], partial(lp_fp, site=hyl, test=amp.test, key=k1),
                limiting_density_range=(-838.8608, 838.8607))

        pri_h_tot = sum(pri_entropy.values())
        pri_life = amp.evaluate_reliability('lifespan', num_samples=1_000_000)


        # Now a comparison of the two
        info_gain = pri_h_tot - pst_h_tot

        print(f'Prior Q90%-HDCR: {pri_life}')
        print(f'Post Q90%-HDCR: {pst_life}')
        print(f'Prior entropy: {pri_h_tot}')
        print(f'Post entropy: {pst_h_tot}')
        print(f'Info gain: {info_gain} nats')


    '''
    ===== 8) Metrics reporting =====
    Generate the regulatory and consumer metrics we can use to market and/or certify the product.
    '''
    # TODO


if __name__ == '__main__':
    idfbcamp_qualification()
