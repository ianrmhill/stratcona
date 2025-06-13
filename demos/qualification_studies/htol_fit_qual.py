# Copyright (c) 2025 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import numpyro as npyro
import numpyro.distributions as dist
from numpyro.distributions.transforms import ComposeTransform, AffineTransform, SoftplusTransform
npyro.set_host_device_count(4)

import jax
import jax.numpy as jnp
import jax.random as rand

import json
import time as time
from functools import partial

import datetime as dt
import certifi
from pymongo import MongoClient
from pymongo.errors import PyMongoError

import seaborn as sb
from matplotlib import pyplot as plt

import gerabaldi
from gerabaldi.models import *

import os
import sys
# This line adds the parent directory to the module search path so that the Stratcona module can be seen and imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import stratcona

DB_NAME = 'stratcona'
COLL_NAME = 'htol-qual'


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


def htol_demo():
    ### Define some constants ###
    boltz_ev = 8.617e-5
    # 125C in Kelvin
    htol_temp, field_temp = 398.15, 328.15
    fail_at_deg_mv = 25

    # Disables and enables for different portions of the study
    run_htol_confidence_analysis = False
    check_model_similarity = False
    run_test_design_analysis = True

    ########################################################################
    ### Define the Arrhenius wear-out model to infer                     ###
    ########################################################################
    # Define the model we will use to fit degradation
    mb = stratcona.SPMBuilder(mdl_name='Arrhenius')

    # Log-scale Arrhenius model
    def l_arrhenius_t(a, eaa, temp):
        temp_coeff = (-eaa) / (boltz_ev * temp)
        return jnp.log(1e9) + jnp.log(a) + temp_coeff

    mb.add_hyperlatent('a_nom', dist.LogNormal, {'loc': 14, 'scale': 0.0001})
    # This linear version is only used for computing entropy to avoid needing to write a log-scale LDDP function
    #mb.add_hyperlatent('a_lin_nom', dist.Normal, {'loc': 14, 'scale': 0.0001})
    pos_scale_tf = ComposeTransform([SoftplusTransform(), AffineTransform(0, 0.1)])
    mb.add_hyperlatent('eaa_nom', dist.Normal, {'loc': 7, 'scale': 0.0001}, pos_scale_tf)
    mb.add_latent('a', 'a_nom')
    mb.add_latent('eaa', 'eaa_nom')

    # Translate a FIT for the chip into a probability of a failed/functional state at time t, can adjust aleatoric
    # uncertainty by scaling the term inside the sigmoid
    def fail_prob(l_fit, t):
        return jax.nn.sigmoid((l_fit - jnp.log(1e9) + jnp.log(t)) * 10)
    mb.add_intermediate('l_fit', l_arrhenius_t)
    mb.add_intermediate('pfail', fail_prob)
    mb.add_observed('failed', dist.Bernoulli, {'probs': 'pfail'}, 1)

    def fail_time(l_fit):
        return jnp.exp(jnp.log(1e9) - l_fit)
    mb.add_predictor('ftime', fail_time)
    amf = stratcona.AnalysisManager(mb.build_model(), rng_seed=6248742)

    ########################################################################
    ### Define the degradation wear-out model to infer
    ########################################################################
    mb = stratcona.SPMBuilder(mdl_name='deg-general')

    # Model provided in JEDEC's JEP122H as generally used NBTI degradation model, equation 5.3.1,
    # log(1000) term to convert to mV
    def dvth_mv(t, l_a, eaa, n, temp):
        return jnp.exp(jnp.log(100) + jnp.log(1000) + l_a + ((-eaa) / (boltz_ev * temp)) + jnp.log(t ** n))

    var_tf = ComposeTransform([SoftplusTransform(), AffineTransform(0, 0.01)])
    pos_scale_tf = ComposeTransform([SoftplusTransform(), AffineTransform(0, 0.1)])
    mb.add_hyperlatent('l_a_nom', dist.Normal, {'loc': 14.5, 'scale': 0.0001})
    mb.add_hyperlatent('l_a_chp', dist.Normal, {'loc': 1.6, 'scale': 0.0001}, var_tf)
    mb.add_hyperlatent('l_a_lot', dist.Normal, {'loc': 2.5, 'scale': 0.0001}, var_tf)
    mb.add_hyperlatent('eaa_nom', dist.Normal, {'loc': 7, 'scale': 0.0001}, pos_scale_tf)
    mb.add_latent('l_a', nom='l_a_nom', chp='l_a_chp', lot='l_a_lot')
    #mb.add_latent('l_a', nom='l_a_nom')
    mb.add_latent('eaa', nom='eaa_nom')

    mb.add_intermediate('degn', dvth_mv)
    mb.add_params(n=0.3, degv=2)
    mb.add_observed('deg', dist.Normal, {'loc': 'degn', 'scale': 'degv'}, 1)

    def fail_thresh(l_a, eaa, n, temp):
        func_args = {'l_a': l_a, 'eaa': eaa, 'n': n, 'temp': temp}

        def residue(time, **kwargs):
            return jnp.abs(fail_at_deg_mv - dvth_mv(time, **kwargs))
        t_life = stratcona.engine.minimization.minimize_jax(residue, func_args, (1, 1e9), precision=1e-4, log_gold=True)
        return t_life
    mb.add_predictor('ftime', fail_thresh)
    amd = stratcona.AnalysisManager(mb.build_model(), rng_seed=833483473)

    # Set the main test and field use conditions for the two models
    htol = stratcona.TestDef('htol', {'htol': {'lot': 3, 'chp': 77}}, {'htol': {'temp': htol_temp, 't': 1000}})
    amf.set_test_definition(htol)
    amd.set_test_definition(htol)
    amf.set_field_use_conditions({'temp': field_temp, 't': 8.865 * 8760})
    amd.set_field_use_conditions({'temp': field_temp, 't': 8.865 * 8760})

    ####################################
    ### Define reliability requirements
    ####################################
    req_lifespan = 8.865 * 8760
    req_q = 99
    metric = stratcona.engine.metrics.qx_lbci
    # The change in Q99%-LBCI shows how ALTs can improve the product lifespan that can be advertised
    rreq = stratcona.ReliabilityRequirement(metric, req_q, req_lifespan)
    amf.relreq, amd.relreq = rreq, rreq

    ####################################
    ### First analysis: how does epistemic uncertainty in Eaa impact reliability prediction in the pass/fail model
    ####################################
    k = rand.key(9292873023)
    # Define the HTOL and field tests with 1 chip and lot
    htols = stratcona.TestDef('htols', {'htols': {'lot': 1, 'chp': 1}}, {'htols': {'temp': htol_temp, 't': 1000}})
    # Set the prior beliefs and evaluate entropy
    amf.relmdl.hyl_beliefs = {'a_nom': {'loc': 13.5, 'scale': 1}, 'eaa_nom': {'loc': 7, 'scale': 0.0001},
                              'a_lin_nom': {'loc': 13.5, 'scale': 1}}
    jax.clear_caches()

    if run_htol_confidence_analysis:
        entropy_samples = 300_000
        def lp_f(vals, site, key, test):
            return amf.relmdl.logp(rng_key=key, test=test, site_vals={site: vals}, conditional=None, dims=(len(vals),))
        hyls = ('a_lin_nom', 'eaa_nom')
        k, k1 = rand.split(k)
        hyl_samples = amf.relmdl.sample_new(k1, htols.dims, htols.conds, (entropy_samples,), keep_sites=hyls)
        pri_entropy = {}
        for hyl in hyls:
            pri_entropy[hyl] = stratcona.engine.bed.entropy(
                hyl_samples[hyl], partial(lp_f, site=hyl, test=htols, key=k1), limiting_density_range=(-838.8608, 838.8607))
        print(f'Prior interpretable entropy: {pri_entropy}')

        # Examine the prior predictive
        k, k1, k2 = rand.split(rand.key(9292873023), 3)
        pri_dist = amf.relmdl.sample_new(k1, amf.field_test.dims, amf.field_test.conds, (1_000_000,),
                                         keep_sites=('field_ftime',), compute_predictors=True)
        pri_htol_dist = amf.relmdl.sample_new(k2, htols.dims, htols.conds, (1_000_000,),
                                              keep_sites=('htols_ftime', 'htols_failed'), compute_predictors=True)
        pri_lifespan = amf.evaluate_reliability('ftime')

        sb.set_theme(style='ticks', font='Times New Roman')
        sb.set_context('notebook')
        fig, p = plt.subplots(2, 1)
        p[0].hist(jnp.log(pri_dist['field_ftime']), 500, density=True, alpha=0.9, color='skyblue', histtype='stepfilled',
                  label='Simulated true lifespan')
        p[1].hist(jnp.log(pri_htol_dist['htols_ftime']), 500, density=True, alpha=0.9, color='skyblue', histtype='stepfilled',
                  label='Simulated HTOL lifespan')

        # Inference on a successful HTOL test result
        y_success = {'htol': {'failed': jnp.zeros((1, 77, 3))}}
        perf = amf.do_inference_is(y_success, n_x=5_000)
        print(perf)
        print(amf.relmdl.hyl_beliefs)
        jax.clear_caches()

        # Examine the posterior predictive
        k, k1, k2 = rand.split(k, 3)
        pst_dist = amf.relmdl.sample_new(k1, amf.field_test.dims, amf.field_test.conds, (1_000_000,),
                                         keep_sites=('field_ftime',), compute_predictors=True)
        pst_htol_dist = amf.relmdl.sample_new(k2, htols.dims, htols.conds, (1_000_000,),
                                              keep_sites=('htols_ftime',), compute_predictors=True)
        pst_lifespan = amf.evaluate_reliability('ftime')

        p[0].hist(jnp.log(pst_dist['field_ftime']), 500, density=True, alpha=0.9, color='darkblue', histtype='stepfilled',
                  label='Simulated true lifespan')
        p[1].hist(jnp.log(pst_htol_dist['htols_ftime']), 500, density=True, alpha=0.9, color='darkblue', histtype='stepfilled',
                  label='Simulated HTOL lifespan')

        # Now repeat, but with a bit of uncertainty as to Eaa
        amf.relmdl.hyl_beliefs = {'a_nom': {'loc': 13.5, 'scale': 0.9}, 'eaa_nom': {'loc': 7, 'scale': 0.05},
                                  'a_lin_nom': {'loc': 13.5, 'scale': 0.9}}
        jax.clear_caches()
        k, k1 = rand.split(k)
        hyl_samples = amf.relmdl.sample_new(k1, htols.dims, htols.conds, (entropy_samples,), keep_sites=hyls)
        pri_entropy = {}
        for hyl in hyls:
            pri_entropy[hyl] = stratcona.engine.bed.entropy(
                hyl_samples[hyl], partial(lp_f, site=hyl, test=htols, key=k1), limiting_density_range=(-838.8608, 838.8607))
        # Target is 10.63
        print(f'Prior interpretable entropy: {pri_entropy}')

        # Examine the prior predictive
        k, k1, k2 = rand.split(k, 3)
        pri_dist = amf.relmdl.sample_new(k1, amf.field_test.dims, amf.field_test.conds, (1_000_000,),
                                         keep_sites=('field_ftime',), compute_predictors=True)
        pri_htol_dist = amf.relmdl.sample_new(k2, htols.dims, htols.conds, (1_000_000,),
                                              keep_sites=('htols_ftime', 'htols_failed'), compute_predictors=True)
        pri_lifespan = amf.evaluate_reliability('ftime')

        p[0].hist(jnp.log(pri_dist['field_ftime']), 500, density=True, alpha=0.5, color='hotpink', histtype='stepfilled',
                  label='Simulated true lifespan')
        p[1].hist(jnp.log(pri_htol_dist['htols_ftime']), 500, density=True, alpha=0.5, color='hotpink', histtype='stepfilled',
                  label='Simulated HTOL lifespan')

        # Inference on a successful HTOL test result
        y_success = {'htol': {'failed': jnp.zeros((1, 77, 3))}}
        perf = amf.do_inference_is(y_success, n_x=5_000)
        print(perf)
        print(amf.relmdl.hyl_beliefs)
        jax.clear_caches()

        # NOTE: The entropy reduction is higher for the second version with Eaa uncertainty, shows how higher IG does
        #       not necessarily imply improved reliability

        # Examine the posterior predictive
        k, k1, k2 = rand.split(k, 3)
        pst_dist = amf.relmdl.sample_new(k1, amf.field_test.dims, amf.field_test.conds, (1_000_000,),
                                         keep_sites=('field_ftime',), compute_predictors=True)
        pst_htol_dist = amf.relmdl.sample_new(k2, htols.dims, htols.conds, (1_000_000,),
                                              keep_sites=('htols_ftime',), compute_predictors=True)
        pst_lifespan = amf.evaluate_reliability('ftime')

        p[0].hist(jnp.log(pst_dist['field_ftime']), 500, density=True, alpha=0.5, color='deeppink', histtype='stepfilled',
                  label='Simulated true lifespan')
        p[1].hist(jnp.log(pst_htol_dist['htols_ftime']), 500, density=True, alpha=0.5, color='deeppink', histtype='stepfilled',
                  label='Simulated HTOL lifespan')

    ####################################
    ### Define the test space
    ####################################
    # Start with the examination of changing the test duration
    half_time = stratcona.TestDef('HTOL500', {'htol': {'lot': 3, 'chp': 77}}, {'htol': {'temp': htol_temp, 't': 500}})
    tenth_time = stratcona.TestDef('HTOL100', {'htol': {'lot': 3, 'chp': 77}}, {'htol': {'temp': htol_temp, 't': 100}})
    dbl_time = stratcona.TestDef('HTOL2000', {'htol': {'lot': 3, 'chp': 77}}, {'htol': {'temp': htol_temp, 't': 2000}})
    test_space = [htol, half_time, tenth_time, dbl_time]

    # Now an examination of reducing the number of chips
    half_chps = stratcona.TestDef('39chpHTOL', {'htol': {'lot': 3, 'chp': 39}}, {'htol': {'temp': htol_temp, 't': 1000}})
    tenth_chps = stratcona.TestDef('8chpHTOL', {'htol': {'lot': 3, 'chp': 8}}, {'htol': {'temp': htol_temp, 't': 1000}})
    test_space.extend([half_chps, tenth_chps])

    # Finally, an analysis of multilevel temperature testing
    temps = [htol_temp - 100, htol_temp - 75, htol_temp - 50, htol_temp - 25, htol_temp - 10, htol_temp, htol_temp + 10, htol_temp + 25]
    temp_combs = []
    for i, t1 in enumerate(temps):
        for j, t2 in enumerate(temps[i+1:]):
            for t3 in temps[i+j+2:]:
                temp_combs.append((t1, t2, t3))
    multilevel_tests = [stratcona.TestDef(
        'multitemp',
        {'t1': {'lot': 3, 'chp': 26}, 't2': {'lot': 3, 'chp': 26}, 't3': {'lot': 3, 'chp': 26}},
        {'t1': {'temp': t1, 't': 1000}, 't2': {'temp': t2, 't': 1000}, 't3': {'temp': t3, 't': 1000}}) for t1, t2, t3 in temp_combs]
    test_space.extend(multilevel_tests)

    ########################################################################
    ### 3. Determine the best experiment to conduct                      ###
    ########################################################################
    # These priors give virtually identical prior predictive distributions, making p(y|d) comparable for both BED runs
    amf.relmdl.hyl_beliefs = {'a_nom': {'loc': 13.2, 'scale': 1.7}, 'eaa_nom': {'loc': 7, 'scale': 0.43}}
    amd.relmdl.hyl_beliefs = {'l_a_nom': {'loc': 13, 'scale': 0.6}, 'eaa_nom': {'loc': 7, 'scale': 0.1},
                              'l_a_chp': {'loc': 4, 'scale': 1.5}, 'l_a_lot': {'loc': 5, 'scale': 1.5}}
    jax.clear_caches()

    ### Model tuning: get the two models to have nearly the same prior predictive distributions
    if check_model_similarity:
        k, k1, k2 = rand.split(k, 3)
        with jax.disable_jit():
            fail_dist = amf.relmdl.sample_new(k1, amf.field_test.dims, amf.field_test.conds, (1_000_000,),
                                              keep_sites=('field_ftime',), compute_predictors=True)
            deg_dist = amd.relmdl.sample_new(k2, amd.field_test.dims, amd.field_test.conds, (1_000_000,),
                                             keep_sites=('field_ftime',), compute_predictors=True)
            fail_lifespan = amf.evaluate_reliability('ftime')
            deg_lifespan = amd.evaluate_reliability('ftime')

        sb.set_theme(style='ticks', font='Times New Roman')
        sb.set_context('notebook')
        fig, p = plt.subplots(1, 1)
        p.hist(jnp.log(fail_dist['field_ftime']), 200, density=True, alpha=0.9, color='hotpink', histtype='stepfilled',
               label='Prior fail lifespan')
        p.hist(jnp.log(deg_dist['field_ftime'].flatten()), 200, density=True, alpha=0.9, color='mediumorchid',
               histtype='stepfilled', label='Prior deg lifespan')
        p.legend()

        plt.show()

    def p_u_func(ig, qx_lbci):
        eig = jnp.sum(ig) / ig.size
        vig = jnp.sum(((ig - eig) ** 2) / ig.size)
        mig = jnp.min(ig)
        rpp = jnp.sum(jnp.where(qx_lbci > req_lifespan, 1, 0)) / qx_lbci.size
        mean_lbci = jnp.sum(qx_lbci) / qx_lbci.size
        return {'eig': eig, 'vig': vig, 'mig': mig, 'rpp': rpp, 'mean-qx-lbci': mean_lbci}

    if run_test_design_analysis:
        dataset = login_to_database()
        n_y_f, n_v_f, n_x_f = 1_000, 1, 1_000
        n_y_d, n_v_d, n_x_d = 50, 250, 500
        #for i in range(len(test_space)):
        for i in range(3):
            k, kf, kd = rand.split(k, 3)
            d = test_space[i]
            sampler = lambda _: d
            u_d_comp_f, perf_stats_f = stratcona.engine.bed.pred_bed_apr25(kf, sampler, 1, n_y_f, n_v_f, n_x_f, amf.relmdl,
                                                                           p_u_func, amf.field_test, 'ftime')
            u_d_comp_d, perf_stats_d = stratcona.engine.bed.pred_bed_apr25(kd, sampler, 1, n_y_d, n_v_d, n_x_d, amd.relmdl,
                                                                           p_u_func, amd.field_test, 'ftime')
            if len(test_space[i].conds.keys()) == 1:
                dims = next(iter(d.dims))
                test_info = {'test-type': 'htol-variant', 'temp': float(d.conds['htol']['temp']), 't': float(d.conds['htol']['t']),
                             'chps': dims.chp, 'lots': dims.lot}
            else:
                test_info = {'test-type': 'multitemp', 't1': d.conds['t1']['temp'], 't2': d.conds['t2']['temp'], 't3': d.conds['t3']['temp']}
            f_entry = {'model': 'fail', 'submit-time': str(dt.datetime.now(tz=dt.UTC)), 'n-y': n_y_f, 'n-x': n_x_f,
                       'eig': float(u_d_comp_f[0]['utility']['eig']), 'vig': float(u_d_comp_f[0]['utility']['vig']),
                       'mig': float(u_d_comp_f[0]['utility']['mig']), 'rpp': float(u_d_comp_f[0]['utility']['rpp']),
                       'mean-qx-lbci': float(u_d_comp_f[0]['utility']['mean-qx-lbci'])} | test_info
            d_entry = {'model': 'deg', 'submit-time': str(dt.datetime.now(tz=dt.UTC)), 'n-y': n_y_d, 'n-v': n_v_d, 'n-x': n_x_d,
                       'eig': float(u_d_comp_d[0]['utility']['eig']), 'vig': float(u_d_comp_d[0]['utility']['vig']),
                       'mig': float(u_d_comp_d[0]['utility']['mig']), 'rpp': float(u_d_comp_d[0]['utility']['rpp']),
                       'mean-qx-lbci': float(u_d_comp_d[0]['utility']['mean-qx-lbci'])} | test_info

            with open(f'../bed_data/htol_bed_evals_f{i}.json', 'w') as f:
                json.dump(f_entry, f)
            with open(f'../bed_data/htol_bed_evals_d{i}.json', 'w') as f:
                json.dump(d_entry, f)
            try_database_upload(dataset, f_entry)
            try_database_upload(dataset, d_entry)

    return


if __name__ == '__main__':
    htol_demo()
