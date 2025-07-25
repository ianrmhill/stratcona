# Copyright (c) 2025 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import time
from functools import partial
import json

import numpyro
import numpyro.distributions as dists
from numpyro.distributions.transforms import ComposeTransform, AffineTransform, SoftplusTransform
# This call has to occur before importing jax
numpyro.set_host_device_count(4)

import jax
import jax.numpy as jnp # noqa: ImportNotAtTopOfFile
import jax.random as rand
import numpy as np

import datetime as dt
import certifi
from pymongo import MongoClient
from pymongo.errors import PyMongoError

import seaborn as sb
from matplotlib import pyplot as plt

import os
import sys
# This line adds the parent directory to the module search path so that the Stratcona module can be seen and imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import stratcona

BOLTZ_EV = 8.617e-5

DB_NAME = 'stratcona'
COLL_NAME = 'em-qual'


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


def electromigration_qualification():
    analyze_prior = False
    run_bed_analysis = False
    examine_utility_measures = False
    viz_sim_models = True
    run_inference = True
    viz_posterior_predictions = True

    '''
    ===== 1) Determine long-term reliability requirements =====
    We will base the requirement on the SIL3 specification of less than 10^-3 (0.001) probability of failure on demand,
    which in the context of long-term reliability we will interpret as a less than 0.1% probability that
    electromigration failures occur before the 10 year intended product useful life. This translates to a metric of a
    worst case 0.1% quantile credible region of at least 10 years.
    '''
    trgt_life = 10 * 8760
    objective = stratcona.ReliabilityRequirement(metric=stratcona.engine.metrics.qx_lbci,
                                                 quantile=99.9, target_lifespan=trgt_life)

    """
    ===== 2) Test resource constraint identification =====
    To determine the bounds of the possible test design space, need to decide on what costs are acceptable. We'll say
    that the reliability team has been given an allocation of 5 chips to use for this qualification testing.

    The maximum and minimum voltages and temperatures that can be used are [0.7, 0.95] and [300, 400] respectively.
    """
    # Since the temperature dependence is already pretty well characterized, the test design space focuses on voltage
    # dependence parameter mean and variability
    test_list, t, volts = [], 400, [0.85, 0.9, 0.95, 1.0, 1.05, 1.1]
    for i, v1 in enumerate(volts):
        for j, v2 in enumerate(volts[i:]):
            for k, v3 in enumerate(volts[i + j:]):
                test_list.append(
                    stratcona.TestDef(
                        'quad-stress',
                        {'v1': {'lot': 1, 'chp': 5}, 'v2': {'lot': 1, 'chp': 5}, 'v3': {'lot': 1, 'chp': 5}},
                        {'v1': {'vdd': v1, 'temp': t, 'n_fins': 240}, 'v2': {'vdd': v2, 'temp': t, 'n_fins': 240},
                         'v3': {'vdd': v3, 'temp': t, 'n_fins': 240}})
                )

    '''
    ===== 3) Model selection =====
    For our inference model we will use the classic electromigration model based on Black's equation. The current
    density is based on the custom IDFBCAMP chip designed by the Ivanov SoC lab.
    
    Physical parameters that are not directly related to wear-out are assumed to be already characterized
    (e.g., threshold voltage) and so are treated as fixed parameters, only the electromigration specific parameters form
    the latent variable space to learn.
    '''
    mb = stratcona.SPMBuilder(mdl_name='Black\'s Electromigration')
    num_devices = 5
    # Wire area in nm^2
    mb.add_params(vth_typ=0.32, i_base=2.8, wire_area=1.024 * 1000 * 1000, k=BOLTZ_EV)

    # Express wire current density as a function of the number of transistors and the voltage applied
    def j_n(n_fins, vdd, vth_typ, i_base):
        return n_fins * i_base * ((vdd - vth_typ) ** 2)

    # The classic model for electromigration failure estimates, DOI: 10.1109/T-ED.1969.16754
    def l_blacks(jn_wire, temp, em_n, em_eaa, wire_area, k):
        return jnp.log(wire_area) - (em_n * jnp.log(jn_wire)) + (em_eaa / (k * temp)) - jnp.log(10 * 3600)

    # Now correspond the degradation mechanisms to the output values
    mb.add_intermediate('jn_wire', j_n)
    mb.add_intermediate('l_fail', l_blacks)
    mb.add_params(l_ttf_var=0.1)
    mb.add_observed('lttf', dists.Normal, {'loc': 'l_fail', 'scale': 'l_ttf_var'}, num_devices)

    var_tf = ComposeTransform([SoftplusTransform(), AffineTransform(0, 0.001)])
    mb.add_hyperlatent('n_nom', dists.Normal, {'loc': 15, 'scale': 2}, AffineTransform(0, 0.1))
    mb.add_hyperlatent('n_dev', dists.Normal, {'loc': 15, 'scale': 3}, var_tf)
    mb.add_hyperlatent('n_chp', dists.Normal, {'loc': 15, 'scale': 3}, var_tf)
    mb.add_hyperlatent('eaa_nom', dists.Normal, {'loc': 40, 'scale': 1}, AffineTransform(0, 0.01))
    mb.add_latent('em_n', 'n_nom', 'n_dev', 'n_chp')
    mb.add_latent('em_eaa', 'eaa_nom')

    # Add the chip-level failure time
    def fail_time(lttf):
        return jnp.exp(jnp.min(lttf))
    mb.add_fail_criterion('lifespan', fail_time)

    am = stratcona.AnalysisManager(mb.build_model(), rel_req=objective, rng_seed=7623842)
    am.set_field_use_conditions({'vdd': 0.85, 'temp': 330, 'n_fins': 48})

    # Can visualize the prior model and see how inference is required to achieve the required predictive confidence.
    if analyze_prior:
        ta1 = stratcona.TestDef('accel1', {'s1': {'lot': 1, 'chp': 1}, 's2': {'lot': 1, 'chp': 1}},
                                {'s1': {'vdd': 0.9, 'temp': 405, 'n_fins': 240}, 's2': {'vdd': 0.9, 'temp': 385, 'n_fins': 240}})

        # Examine the prior predictive
        with jax.disable_jit():
            k, k1, k2 = rand.split(rand.key(9292873023), 3)
            pri_dist = am.relmdl.sample_new(k1, am.field_test.dims, am.field_test.conds, (1_000_000,),
                                            keep_sites=('field_lttf',), compute_predictors=True)
            pri_accel_dist = am.relmdl.sample_new(k2, ta1.dims, ta1.conds, (1_000_000,),
                                                  keep_sites=('s1_lttf', 's2_lttf'), compute_predictors=True)
            pri_lifespan = am.evaluate_reliability('lifespan')

        sb.set_theme(style='ticks', font='Times New Roman')
        sb.set_context('notebook')
        fig, p = plt.subplots(2, 1)
        p[0].hist(pri_dist['field_lttf'].flatten(), 200, density=True, alpha=0.9, color='skyblue', histtype='stepfilled',
                  label='Simulated true lifespan')
        p[1].hist(pri_accel_dist['s1_lttf'].flatten(), 200, density=True, alpha=0.9, color='skyblue', histtype='stepfilled',
                  label='Simulated HTOL lifespan')
        p[1].hist(pri_accel_dist['s2_lttf'].flatten(), 200, density=True, alpha=0.9, color='hotpink', histtype='stepfilled',
                  label='Simulated HTOL lifespan')

        plt.show()

    '''
    ===== 4) Accelerated test design analysis =====
    Once we have BED statistics on all the possible tests we perform our analysis to determine which one to use.
    '''
    # Reset the prior beliefs to be sure they are correct for the BED analysis phase
    am.relmdl.hyl_beliefs = {'n_nom': {'loc': 15, 'scale': 2}, 'n_dev': {'loc': 15, 'scale': 3}, 'n_chp': {'loc': 15, 'scale': 3}, 'eaa_nom': {'loc': 40, 'scale': 1}}
    jax.clear_caches()

    if run_bed_analysis:
        def em_u_func(ig, qx_lbci, test_duration):
            eig = jnp.sum(ig) / ig.size
            vig = jnp.sum(((ig - eig) ** 2) / ig.size)
            mig = jnp.min(ig)
            rpp = jnp.sum(jnp.where(qx_lbci > trgt_life, 1, 0)) / qx_lbci.size
            e_lbci = jnp.sum(qx_lbci) / qx_lbci.size
            etime = jnp.sum(test_duration) / test_duration.size
            maxtime = jnp.max(test_duration)
            return {'eig': eig, 'vig': vig, 'mig': mig, 'rpp': rpp, 'e-qx-lbci': e_lbci,
                    'etime': etime, 'maxtime': maxtime}

        dataset = login_to_database()
        n_y, n_v, n_x = 30, 100, 200
        batches = 1 # 8
        d_batch_size = 7
        d_samplers = [stratcona.assistants.iter_sampler(test_list[i*d_batch_size:(i*d_batch_size)+d_batch_size]) for i in range(batches)]
        keys = rand.split(am._derive_key(), batches)
        eval_d_batch_p = partial(stratcona.engine.bed.pred_bed_apr25, n_d=d_batch_size, n_y=n_y, n_v=n_v, n_x=n_x,
                                 spm=am.relmdl, utility=em_u_func, field_d=am.field_test, predictor='lifespan')

        for i in range(batches):
            udc, _ = eval_d_batch_p(keys[i], d_samplers[i])

            simplified = {'batch': i, 'batch-size': d_batch_size, 'submit-time': str(dt.datetime.now(tz=dt.UTC)),
                          'n-y': n_y, 'n-v': n_v, 'n-x': n_x}
            for j in range(d_batch_size):
                simplified[f'{str(i)}-{str(j)}'] = {
                    'v1': float(udc[j]['design'].conds['v1']['vdd']), 'v2': float(udc[j]['design'].conds['v2']['vdd']), 'v3': float(udc[j]['design'].conds['v3']['vdd']),
                    'eig': float(udc[j]['utility']['eig']), 'vig': float(udc[j]['utility']['vig']), 'mig': float(udc[j]['utility']['mig']),
                    'rpp': float(udc[j]['utility']['rpp']), 'e-qx-lbci': float(udc[j]['utility']['e-qx-lbci']),
                    'e-logtime': float(udc[j]['utility']['etime']), 'max-logtime': float(udc[j]['utility']['maxtime'])}
            with open(f'../bed_data/em_bed_evals_batch{i}.json', 'w') as f:
                json.dump(simplified, f)
            try_database_upload(dataset, simplified)

    if examine_utility_measures:
        # Connect to the weartest database
        tls_ca = certifi.where()
        uri = "mongodb+srv://arbutus.6v6mkhr.mongodb.net/?authSource=%24external&authMechanism=MONGODB-X509&retryWrites=true&w=majority"
        mongo_client = MongoClient(uri, tls=True, tlsCertificatekeyFile='../cert/mongo_cert.pem', tlsCAFile=tls_ca)
        db = mongo_client['stratcona']
        try:
            mongo_client.admin.command('ping')
        except PyMongoError as e:
            print(e)
            print("\nCould not connect to database successfully, exiting...")
            exit()

        dset_1 = db[COLL_NAME]
        batches = dset_1.find()

        x, y, z = [], [], []
        rpp, elbci, eig, vig, mig, eltime, maxltime = [], [], [], [], [], [], []
        for i, b in enumerate(batches):
            for j in range(7):
                d = b[f'{i}-{j}']
                v1, v2, v3 = d['v1'], d['v2'], d['v3']
                x.append(v1)
                y.append(v2)
                z.append(v3)
                rpp.append(d['rpp'])
                elbci.append(d['e-qx-lbci'])
                eig.append(d['eig'])
                vig.append(d['vig'])
                mig.append(d['mig'])
                eltime.append(d['e-logtime'])
                maxltime.append(d['max-logtime'])

        rpp_max = np.argmax(rpp)
        rpp_min = np.argmin(rpp)
        rpp_mean = np.mean(rpp)
        elbci_max = np.argmax(elbci)
        eig_max = np.argmax(eig)
        vig_max = np.argmax(vig)
        mig_max = np.argmax(mig)
        eltime_max = np.argmax(eltime)
        eltime_min = np.argmin(eltime)
        maxltime_max = np.argmax(maxltime)

        # Evaluate utility for each design to select one
        def u(rpp, eltime):
            return (4000 * (rpp - 0.278)) + (533 - np.exp(eltime))

        d_u = u(np.array(rpp), np.array(eltime))
        u_max = np.argmax(d_u)

        # Generate the 3D plot
        sb.set_theme(style='ticks', font='Times New Roman')
        sb.set_context('talk')
        plt.rcParams['mathtext.fontset'] = 'custom'
        plt.rcParams['mathtext.rm'] = 'Times New Roman'
        plt.rcParams['mathtext.it'] = 'Times New Roman'
        plt.rcParams['font.family'] = 'Times New Roman'

        fig = plt.figure()

        p = fig.add_subplot(1, 2, 1, projection='3d')
        vals = p.scatter(x, y, zs=z, c=rpp, cmap='cividis', s=100, vmin=0.0, vmax=0.6)
        fig.colorbar(vals, ax=p, orientation='vertical', label='Requirement Pass Probability')
        p.set_xlabel('Sub-test voltage (V)', labelpad=10)
        p.set_ylabel('Sub-test voltage (V)', labelpad=10)
        p.set_zlabel('Sub-test voltage (V)', labelpad=10)

        p2 = fig.add_subplot(1, 2, 2, projection='3d')
        vals = p2.scatter(x, y, zs=z, c=eltime, cmap='cividis', s=100, vmin=np.log(300), vmax=np.log(3000))
        cbar2 = fig.colorbar(vals, ax=p2, orientation='vertical', label='Expected test time (log hours)',
                             ticks=[np.log(300), np.log(650), np.log(1000), np.log(2000), np.log(3000)])
        cbar2.ax.set_yticklabels(['300', '650', '1000', '2000', '3000'])
        p2.set_xlabel('Sub-test voltage (V)', labelpad=10)
        p2.set_ylabel('Sub-test voltage (V)', labelpad=10)
        p2.set_zlabel('Sub-test voltage (V)', labelpad=10)

        fig.subplots_adjust(hspace=0.5)

        plt.show()

    '''
    ===== 5) Conduct the selected test =====
    Here we use Gerabaldi to emulate running the test on a real-world product. The simulator only works with
    temporal models, thus our electromigration failures are determined not by Black's equation but by an entirely
    different custom probabilistic model.
    
    Optional fallbacks to Black's equation or manual entry failure data are provided.
    '''
    am.set_test_definition(test_list[55])
    kpri, k1, k2, k3 = rand.split(am._derive_key(), 4)

    # Reset the prior beliefs to be sure they are correct for the BED analysis phase
    am.relmdl.hyl_beliefs = {'n_nom': {'loc': 15, 'scale': 2}, 'n_dev': {'loc': 15, 'scale': 3},
                             'n_chp': {'loc': 15, 'scale': 3}, 'eaa_nom': {'loc': 40, 'scale': 1}}
    jax.clear_caches()
    pri_dist = am.relmdl.sample_new(kpri, am.field_test.dims, am.field_test.conds, (1_000_000,),
                                    keep_sites=('field_lifespan',), compute_predictors=True)
    pri_lifespan = am.evaluate_reliability('lifespan')

    # Poor reliability
    am.relmdl.hyl_beliefs = {'n_nom': {'loc': 15, 'scale': 0.0001}, 'n_dev': {'loc': 15, 'scale': 0.0001},
                             'n_chp': {'loc': 15, 'scale': 0.0001}, 'eaa_nom': {'loc': 39, 'scale': 0.0001}}
    jax.clear_caches()
    sd1 = am.sim_test_meas_new()
    sd1_inf = {'v1': {'lttf': sd1['v1_lttf']}, 'v2': {'lttf': sd1['v2_lttf']}, 'v3': {'lttf': sd1['v3_lttf']}}
    if viz_sim_models:
        s1_dist = am.relmdl.sample_new(k1, am.field_test.dims, am.field_test.conds, (1_000_000,),
                                        keep_sites=('field_lifespan',), compute_predictors=True)
        s1_lifespan = am.evaluate_reliability('lifespan')

    # Barely acceptable reliability
    am.relmdl.hyl_beliefs = {'n_nom': {'loc': 15, 'scale': 0.0001}, 'n_dev': {'loc': 17, 'scale': 0.0001},
                             'n_chp': {'loc': 16, 'scale': 0.0001}, 'eaa_nom': {'loc': 40, 'scale': 0.0001}}
    jax.clear_caches()
    sd2 = am.sim_test_meas_new()
    sd2_inf = {'v1': {'lttf': sd2['v1_lttf']}, 'v2': {'lttf': sd2['v2_lttf']}, 'v3': {'lttf': sd2['v3_lttf']}}
    if viz_sim_models:
        s2_dist = am.relmdl.sample_new(k2, am.field_test.dims, am.field_test.conds, (1_000_000,),
                                        keep_sites=('field_lifespan',), compute_predictors=True)
        s2_lifespan = am.evaluate_reliability('lifespan')

    # Good reliability
    am.relmdl.hyl_beliefs = {'n_nom': {'loc': 12.9, 'scale': 0.0001}, 'n_dev': {'loc': 11, 'scale': 0.0001},
                             'n_chp': {'loc': 12, 'scale': 0.0001}, 'eaa_nom': {'loc': 40, 'scale': 0.0001}}
    jax.clear_caches()
    sd3 = am.sim_test_meas_new()
    sd3_inf = {'v1': {'lttf': sd3['v1_lttf']}, 'v2': {'lttf': sd3['v2_lttf']}, 'v3': {'lttf': sd3['v3_lttf']}}
    if viz_sim_models:
        s3_dist = am.relmdl.sample_new(k3, am.field_test.dims, am.field_test.conds, (1_000_000,),
                                        keep_sites=('field_lifespan',), compute_predictors=True)
        s3_lifespan = am.evaluate_reliability('lifespan')

    # Check the data
    if viz_sim_models:
        # Examine the prior predictive
        sb.set_theme(style='ticks', font='Times New Roman')
        sb.set_context('talk')
        plt.rcParams['mathtext.fontset'] = 'custom'
        plt.rcParams['mathtext.rm'] = 'Times New Roman'
        plt.rcParams['mathtext.it'] = 'Times New Roman'
        plt.rcParams['font.family'] = 'Times New Roman'

        fig, p = plt.subplots(1, 1)
        p.axvline(jnp.log(87600), 0, 1, color='black', linestyle='dashed',
                  label=f'Required Q99.9%-LBCI: 10 years')

        p.hist(jnp.log(pri_dist['field_lifespan']), 200, density=True, alpha=0.9, color='skyblue', histtype='stepfilled',
               label='Prior predictive distribution')
        p.axvline(jnp.log(pri_lifespan), 0, 1, color='skyblue', linestyle='dashed',
                  label=f'Prior Q99.9%-LBCI: {round(float(pri_lifespan / 8760), 2)} years')

        p.hist(jnp.log(s1_dist['field_lifespan']), 100, density=True, alpha=0.6, color='maroon', histtype='stepfilled',
               label='Poor true predictive distribution')
        p.axvline(jnp.log(s1_lifespan), 0, 1, color='maroon', linestyle='dashed',
                  label=f'Poor true Q99.9%-LBCI: {round(float(s1_lifespan / 8760), 2)} years')

        p.hist(jnp.log(s2_dist['field_lifespan']), 100, density=True, alpha=0.6, color='goldenrod', histtype='stepfilled',
               label='Barely true predictive distribution')
        p.axvline(jnp.log(s2_lifespan), 0, 1, color='goldenrod', linestyle='dashed',
                  label=f'Barely true Q99.9%-LBCI: {round(float(s2_lifespan / 8760), 2)} years')

        p.hist(jnp.log(s3_dist['field_lifespan']), 100, density=True, alpha=0.6, color='olivedrab', histtype='stepfilled',
               label='High true predictive distribution')
        p.axvline(jnp.log(s3_lifespan), 0, 1, color='olivedrab', linestyle='dashed',
                  label=f'High true Q99.9%-LBCI: {round(float(s3_lifespan / 8760), 2)} years')

        p.set_xlim(9, 15)
        p.set_xticks([jnp.log(8760), jnp.log(26280), jnp.log(87600), jnp.log(262800), jnp.log(876000)],
                     ['1', '3', '10', '30', '100'])
        p.set_xlabel('Predicted Lifespan (log years)')
        p.set_ylabel('Probability Density')
        p.legend(loc='upper right', fontsize='small')
        fig.subplots_adjust(bottom=0.3)

    '''
    ===== 6) Perform model inference =====
    Update our model based on the emulated test data. First need to extract the failure times from the measurements of
    fail states.
    '''
    k1, k2, k3 = rand.split(am._derive_key(), 3)

    if run_inference:
        # Reset the prior beliefs
        am.relmdl.hyl_beliefs = {'n_nom': {'loc': 15, 'scale': 2}, 'n_dev': {'loc': 15, 'scale': 3},
                                 'n_chp': {'loc': 15, 'scale': 3}, 'eaa_nom': {'loc': 40, 'scale': 1}}
        jax.clear_caches()
        start_time = time.time()
        am.do_inference(sd1_inf)
        print(f'Inference time taken: {time.time() - start_time}')
        jax.clear_caches()
    pst1_dist = am.relmdl.sample_new(k1, am.field_test.dims, am.field_test.conds, (1_000_000,),
                                     keep_sites=('field_lifespan',), compute_predictors=True)
    pst1_lifespan = am.evaluate_reliability('lifespan')

    if run_inference:
        # Reset the prior beliefs
        am.relmdl.hyl_beliefs = {'n_nom': {'loc': 15, 'scale': 2}, 'n_dev': {'loc': 15, 'scale': 3},
                                 'n_chp': {'loc': 15, 'scale': 3}, 'eaa_nom': {'loc': 40, 'scale': 1}}
        jax.clear_caches()
        start_time = time.time()
        am.do_inference(sd2_inf)
        print(f'Inference time taken: {time.time() - start_time}')
        jax.clear_caches()
    pst2_dist = am.relmdl.sample_new(k2, am.field_test.dims, am.field_test.conds, (1_000_000,),
                                     keep_sites=('field_lifespan',), compute_predictors=True)
    pst2_lifespan = am.evaluate_reliability('lifespan')

    if run_inference:
        # Reset the prior beliefs
        am.relmdl.hyl_beliefs = {'n_nom': {'loc': 15, 'scale': 2}, 'n_dev': {'loc': 15, 'scale': 3},
                                 'n_chp': {'loc': 15, 'scale': 3}, 'eaa_nom': {'loc': 40, 'scale': 1}}
        jax.clear_caches()
        start_time = time.time()
        am.do_inference(sd3_inf)
        print(f'Inference time taken: {time.time() - start_time}')
        jax.clear_caches()
    pst3_dist = am.relmdl.sample_new(k3, am.field_test.dims, am.field_test.conds, (1_000_000,),
                                     keep_sites=('field_lifespan',), compute_predictors=True)
    pst3_lifespan = am.evaluate_reliability('lifespan')

    '''
    ===== 7) Examine predicted reliability and metric reporting =====
    Check whether we meet the long-term reliability target and with sufficient confidence.
    '''

    if viz_posterior_predictions:
        # Examine the prior predictive
        sb.set_theme(style='ticks', font='Times New Roman')
        sb.set_context('talk')
        plt.rcParams['mathtext.fontset'] = 'custom'
        plt.rcParams['mathtext.rm'] = 'Times New Roman'
        plt.rcParams['mathtext.it'] = 'Times New Roman'
        plt.rcParams['font.family'] = 'Times New Roman'

        fig, p = plt.subplots(1, 1)
        p.axvline(jnp.log(87600), 0, 1, color='black', linestyle='dashed',
                  label=f'Required Q99.9%-LBCI: 10 years')

        p.hist(jnp.log(pri_dist['field_lifespan']), 200, density=True, alpha=0.9, color='skyblue', histtype='stepfilled',
               label='Prior predictive distribution')
        p.axvline(jnp.log(pri_lifespan), 0, 1, color='skyblue', linestyle='dashed',
                  label=f'Prior Q99.9%-LBCI: {round(float(pri_lifespan / 8760), 2)} years')

        p.hist(jnp.log(pst1_dist['field_lifespan']), 200, density=True, alpha=0.6, color='maroon', histtype='stepfilled',
               label='Poor posterior predictive distribution')
        p.axvline(jnp.log(pst1_lifespan), 0, 1, color='maroon', linestyle='dashed',
                  label=f'Poor posterior Q99.9%-LBCI: {round(float(pst1_lifespan / 8760), 2)} years')

        p.hist(jnp.log(pst2_dist['field_lifespan']), 200, density=True, alpha=0.6, color='goldenrod', histtype='stepfilled',
               label='Barely posterior predictive distribution')
        p.axvline(jnp.log(pst2_lifespan), 0, 1, color='goldenrod', linestyle='dashed',
                  label=f'Barely posterior Q99.9%-LBCI: {round(float(pst2_lifespan / 8760), 2)} years')

        p.hist(jnp.log(pst3_dist['field_lifespan']), 200, density=True, alpha=0.6, color='olivedrab', histtype='stepfilled',
               label='High posterior predictive distribution')
        p.axvline(jnp.log(pst3_lifespan), 0, 1, color='olivedrab', linestyle='dashed',
                  label=f'High posterior Q99.9%-LBCI: {round(float(pst3_lifespan / 8760), 2)} years')

        p.set_xlim(9, 15)
        p.set_xticks([jnp.log(8760), jnp.log(26280), jnp.log(87600), jnp.log(262800), jnp.log(876000)],
                     ['1', '3', '10', '30', '100'])
        p.set_xlabel('Predicted Lifespan (log years)')
        p.set_ylabel('Probability Density')
        p.legend(loc='upper right', fontsize='small')
        fig.subplots_adjust(bottom=0.3)

    plt.show()

    '''
    ===== 8) Metrics reporting =====
    Generate the regulatory and consumer metrics we can use to market and/or certify the product.
    '''
    # TODO


if __name__ == '__main__':
    electromigration_qualification()
