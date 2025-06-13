# Copyright (c) 2025 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import numpyro as npyro
import numpyro.distributions as dists
from numpyro.distributions.transforms import ComposeTransform, AffineTransform, SoftplusTransform
# Device count has to be set before importing jax
npyro.set_host_device_count(4)

import jax
import jax.numpy as jnp
import jax.random as rand

import time
from functools import partial
import json
import pandas as pd
from io import StringIO

import seaborn as sb
from matplotlib import pyplot as plt

import os
import sys
# This line adds the parent directory to the module search path so that the Stratcona module can be seen and imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import stratcona

BOLTZ_EV = 8.617e-5
CELSIUS_TO_KELVIN = 273.15


# Working!!!
def simple_high_dev_count():
    # Define the simple model
    mb = stratcona.SPMBuilder('barebones')
    var_tf = dists.transforms.ComposeTransform([dists.transforms.SoftplusTransform(), dists.transforms.AffineTransform(0, 0.1)])
    mb.add_hyperlatent('x', dists.Normal, {'loc': 1.3, 'scale': 0.0001})
    mb.add_hyperlatent('xs', dists.Normal, {'loc': 1, 'scale': 0.0001}, var_tf)
    mb.add_latent('v', nom='x', dev='xs')
    mb.add_params(ys=0.04)
    mb.add_observed('y', dists.Normal, {'loc': 'v', 'scale': 'ys'}, 100)
    am = stratcona.AnalysisManager(mb.build_model(), rng_seed=48)

    # Set up the test and sample observations
    d = stratcona.TestDef('bare', {'e': {'lot': 1, 'chp': 1}}, {'e': {}})
    am.set_test_definition(d)
    k = rand.key(3737)
    y_s = am.relmdl.sample_new(k, d.dims, d.conds, (), am.relmdl.observes)
    y = {'e': {'y': y_s['e_y']}}
    print(f'Mean - {jnp.mean(y_s["e_y"])}, dev - {jnp.std(y_s["e_y"])}')

    # Perform inference using custom importance sampling with the v resampling procedure
    am.relmdl.hyl_beliefs = {'x': {'loc': 1.2, 'scale': 0.2}, 'xs': {'loc': 0.9, 'scale': 0.2}}
    jax.clear_caches()
    perf = am.do_inference_is(y, n_x=1000)
    print(perf)
    print(am.relmdl.hyl_beliefs)

    # Now compare to HMC
    am.relmdl.hyl_beliefs = {'x': {'loc': 1.2, 'scale': 0.2}, 'xs': {'loc': 0.9, 'scale': 0.2}}
    jax.clear_caches()
    am.do_inference(y)
    print(am.relmdl.hyl_beliefs)


# Working!!!
def simple_chip_level():
    # Define the simple model
    mb = stratcona.SPMBuilder('barebones')
    var_tf = dists.transforms.ComposeTransform([dists.transforms.SoftplusTransform(), dists.transforms.AffineTransform(0, 0.1)])
    mb.add_hyperlatent('x', dists.Normal, {'loc': 6.8, 'scale': 0.0001})
    mb.add_hyperlatent('xsd', dists.Normal, {'loc': 3, 'scale': 0.0001}, var_tf)
    mb.add_hyperlatent('xsc', dists.Normal, {'loc': 6, 'scale': 0.0001}, var_tf)
    mb.add_latent('v', nom='x', dev='xsd', chp='xsc')
    mb.add_params(ys=0.1)
    mb.add_observed('y', dists.Normal, {'loc': 'v', 'scale': 'ys'}, 5)
    am = stratcona.AnalysisManager(mb.build_model(), rng_seed=48)

    # Set up the test and sample observations
    d = stratcona.TestDef('bare', {'e': {'lot': 1, 'chp': 20}}, {'e': {}})
    am.set_test_definition(d)
    k = rand.key(9784)
    y_s = am.relmdl.sample_new(k, d.dims, d.conds, (), am.relmdl.observes)
    y = {'e': {'y': y_s['e_y']}}
    print(f'Mean - {jnp.mean(y_s["e_y"])}, dev - {jnp.std(y_s["e_y"])}')

    # Perform inference using custom importance sampling with the v resampling procedure
    am.relmdl.hyl_beliefs = {'x': {'loc': 6, 'scale': 1},
                             'xsd': {'loc': 4, 'scale': 2}, 'xsc': {'loc': 4, 'scale': 2}}
    jax.clear_caches()
    perf = am.do_inference_is(y, n_x=3_000, n_v=500)
    print(perf)
    print(am.relmdl.hyl_beliefs)

    # Now compare to HMC
    am.relmdl.hyl_beliefs = {'x': {'loc': 6, 'scale': 1},
                             'xsd': {'loc': 4, 'scale': 2}, 'xsc': {'loc': 4, 'scale': 2}}
    jax.clear_caches()
    am.do_inference(y)
    print(am.relmdl.hyl_beliefs)


# Working!!!
def simple_all_level():
    # Define the simple model
    mb = stratcona.SPMBuilder('barebones')
    var_tf = dists.transforms.ComposeTransform([dists.transforms.SoftplusTransform(), dists.transforms.AffineTransform(0, 0.1)])
    mb.add_hyperlatent('x', dists.Normal, {'loc': 6.8, 'scale': 0.0001})
    mb.add_hyperlatent('xsd', dists.Normal, {'loc': 3, 'scale': 0.0001}, var_tf)
    mb.add_hyperlatent('xsc', dists.Normal, {'loc': 6, 'scale': 0.0001}, var_tf)
    mb.add_hyperlatent('xsl', dists.Normal, {'loc': 3, 'scale': 0.0001}, var_tf)
    mb.add_latent('v', nom='x', dev='xsd', chp='xsc', lot='xsl')
    mb.add_params(ys=0.1)
    mb.add_observed('y', dists.Normal, {'loc': 'v', 'scale': 'ys'}, 4)
    am = stratcona.AnalysisManager(mb.build_model(), rng_seed=48)

    # Set up the test and sample observations
    d = stratcona.TestDef('bare', {'e': {'lot': 5, 'chp': 5}}, {'e': {}})
    am.set_test_definition(d)
    k = rand.key(6536)
    y_s = am.relmdl.sample_new(k, d.dims, d.conds, (), am.relmdl.observes)
    y = {'e': {'y': y_s['e_y']}}
    print(f'Mean - {jnp.mean(y_s["e_y"])}, dev - {jnp.std(y_s["e_y"])}')

    # Perform inference using custom importance sampling with the v resampling procedure
    am.relmdl.hyl_beliefs = {'x': {'loc': 6, 'scale': 1}, 'xsd': {'loc': 4, 'scale': 1},
                             'xsc': {'loc': 5, 'scale': 1}, 'xsl': {'loc': 4, 'scale': 1}}
    jax.clear_caches()
    perf = am.do_inference_is(y, n_x=3_000, n_v=500)
    print(perf)
    print(am.relmdl.hyl_beliefs)

    # Now compare to HMC
    am.relmdl.hyl_beliefs = {'x': {'loc': 6, 'scale': 1}, 'xsd': {'loc': 4, 'scale': 1},
                             'xsc': {'loc': 5, 'scale': 1}, 'xsl': {'loc': 4, 'scale': 1}}
    jax.clear_caches()
    am.do_inference(y)
    print(am.relmdl.hyl_beliefs)


# Working!!!
def linear_dev_var():
    mb = stratcona.SPMBuilder('linear')
    var_tf = dists.transforms.ComposeTransform([dists.transforms.SoftplusTransform(), dists.transforms.AffineTransform(0, 0.1)])
    mb.add_hyperlatent('mn', dists.Normal, {'loc': 1.3, 'scale': 0.0001})
    mb.add_hyperlatent('bn', dists.Normal, {'loc': 5, 'scale': 0.0001})
    mb.add_hyperlatent('bd', dists.Normal, {'loc': 8, 'scale': 0.0001}, var_tf)
    mb.add_latent('m', nom='mn')
    mb.add_latent('b', nom='bn', dev='bd')
    def linear(m, b, x): return (m * x) + b
    mb.add_intermediate('yn', linear)
    mb.add_params(ys=0.1)
    mb.add_observed('y', dists.Normal, {'loc': 'yn', 'scale': 'ys'}, 10)
    am = stratcona.AnalysisManager(mb.build_model(), rng_seed=64234)

    # Set up the test and sample observations
    d = stratcona.TestDef('multi_x',
                          {'p1': {'lot': 1, 'chp': 1}, 'p2': {'lot': 1, 'chp': 1}, 'p3': {'lot': 1, 'chp': 1}},
                          {'p1': {'x': 0}, 'p2': {'x': 4}, 'p3': {'x': 9}})
    am.set_test_definition(d)
    k = rand.key(3737)
    y_s = am.relmdl.sample_new(k, d.dims, d.conds, (), am.relmdl.observes)
    y = {'p1': {'y': y_s['p1_y']}, 'p2': {'y': y_s['p2_y']}, 'p3': {'y': y_s['p3_y']}}

    # Perform inference using custom importance sampling with the v resampling procedure
    am.relmdl.hyl_beliefs = {'mn': {'loc': 1.2, 'scale': 0.3},
                             'bn': {'loc': 4.8, 'scale': 0.5}, 'bd': {'loc': 7.8, 'scale': 0.4}}
    jax.clear_caches()
    perf = am.do_inference_is(y, n_x=10_000)
    print(perf)
    print(am.relmdl.hyl_beliefs)

    # Now compare to HMC
    am.relmdl.hyl_beliefs = {'mn': {'loc': 1.2, 'scale': 0.3},
                             'bn': {'loc': 4.8, 'scale': 0.5}, 'bd': {'loc': 7.8, 'scale': 0.4}}
    jax.clear_caches()
    am.do_inference(y)
    print(am.relmdl.hyl_beliefs)


# Working!!!
def loglin_multilevel_var():
    # Introduces lognormal hyls, transforms, and more complex behaviour
    boltz_ev = 8.617e-5
    # Define the model we will use to fit degradation
    mb = stratcona.SPMBuilder(mdl_name='Arrhenius')
    # Log-scale Arrhenius model
    def l_arrhenius_t(l_a, eaa, temp):
        temp_coeff = (-eaa) / (boltz_ev * temp)
        return jnp.log(1e9) + l_a + temp_coeff

    mb.add_hyperlatent('l_a_nom', dists.Normal, {'loc': 14, 'scale': 0.0001})
    pos_scale_tf = ComposeTransform([SoftplusTransform(), AffineTransform(0, 0.1)])
    mb.add_hyperlatent('l_a_chp', dists.Normal, {'loc': 3, 'scale': 0.0001}, pos_scale_tf)
    mb.add_hyperlatent('l_a_lot', dists.Normal, {'loc': 4, 'scale': 0.0001}, pos_scale_tf)
    mb.add_hyperlatent('eaa_nom', dists.Normal, {'loc': 7, 'scale': 0.0001}, pos_scale_tf)

    mb.add_latent('l_a', 'l_a_nom', chp='l_a_chp', lot='l_a_lot')
    mb.add_latent('eaa', 'eaa_nom')
    mb.add_intermediate('fitn', l_arrhenius_t)
    mb.add_params(fitv=0.2)
    mb.add_observed('fit', dists.Normal, {'loc': 'fitn', 'scale': 'fitv'}, 1)
    am = stratcona.AnalysisManager(mb.build_model(), rng_seed=52372)

    # Set up the test and sample observations
    d = stratcona.TestDef('multi_temp',
                          {'t1': {'lot': 5, 'chp': 5}, 't2': {'lot': 5, 'chp': 5}, 't3': {'lot': 5, 'chp': 5}},
                          {'t1': {'temp': 300}, 't2': {'temp': 350}, 't3': {'temp': 400}})
    am.set_test_definition(d)
    k = rand.key(83434)
    y_s = am.relmdl.sample_new(k, d.dims, d.conds, (), am.relmdl.observes)
    y = {'t1': {'fit': y_s['t1_fit']}, 't2': {'fit': y_s['t2_fit']}, 't3': {'fit': y_s['t3_fit']}}

    # Perform inference using custom importance sampling with the v resampling procedure
    am.relmdl.hyl_beliefs = {'l_a_nom': {'loc': 13, 'scale': 1.5}, 'eaa_nom': {'loc': 7, 'scale': 0.3},
                             'l_a_chp': {'loc': 5, 'scale': 2}, 'l_a_lot': {'loc': 5, 'scale': 2}}
    jax.clear_caches()
    perf = am.do_inference_is(y, n_x=4_000)
    print(perf)
    print(am.relmdl.hyl_beliefs)

    # Now compare to HMC
    am.relmdl.hyl_beliefs = {'l_a_nom': {'loc': 13, 'scale': 1.5}, 'eaa_nom': {'loc': 7, 'scale': 0.3},
                             'l_a_chp': {'loc': 5, 'scale': 2}, 'l_a_lot': {'loc': 5, 'scale': 2}}
    jax.clear_caches()
    am.do_inference(y)
    print(am.relmdl.hyl_beliefs)


# Working!!!
def loglin_bernoulli_obs_no_var():
    # Introduces lognormal hyls, transforms, and more complex behaviour
    boltz_ev = 8.617e-5
    # Define the model we will use to fit degradation
    mb = stratcona.SPMBuilder(mdl_name='Arrhenius')

    # Log-scale Arrhenius model
    def l_arrhenius_t(l_a, eaa, temp):
        temp_coeff = (-eaa) / (boltz_ev * temp)
        return jnp.log(1e9) + l_a + temp_coeff

    def fail_prob(l_fit, t):
        return jax.nn.sigmoid(l_fit - jnp.log(1e9) + jnp.log(t))

    mb.add_hyperlatent('l_a_nom', dists.Normal, {'loc': 14, 'scale': 0.0001})
    pos_scale_tf = ComposeTransform([SoftplusTransform(), AffineTransform(0, 0.1)])
    mb.add_hyperlatent('eaa_nom', dists.Normal, {'loc': 7, 'scale': 0.0001}, pos_scale_tf)

    mb.add_latent('l_a', 'l_a_nom')
    mb.add_latent('eaa', 'eaa_nom')
    mb.add_intermediate('l_fit', l_arrhenius_t)
    mb.add_intermediate('pfail', fail_prob)
    mb.add_observed('failed', dists.Bernoulli, {'probs': 'pfail'}, 1)
    am = stratcona.AnalysisManager(mb.build_model(), rng_seed=6248742)

    # Set up the test and sample observations
    d = stratcona.TestDef('multi_temp',
                          {'t1': {'lot': 2, 'chp': 2}, 't2': {'lot': 2, 'chp': 2}, 't3': {'lot': 2, 'chp': 2}},
                          {'t1': {'temp': 300, 't': 1000}, 't2': {'temp': 350, 't': 1000},
                           't3': {'temp': 400, 't': 1000}})
    am.set_test_definition(d)
    k = rand.key(83434)
    y_s = am.relmdl.sample_new(k, d.dims, d.conds, (), am.relmdl.observes)
    y = {'t1': {'failed': y_s['t1_failed']}, 't2': {'failed': y_s['t2_failed']}, 't3': {'failed': y_s['t3_failed']}}
    print(y)

    # Perform inference using custom importance sampling with the v resampling procedure
    am.relmdl.hyl_beliefs = {'l_a_nom': {'loc': 13.5, 'scale': 1}, 'eaa_nom': {'loc': 7, 'scale': 0.3}}
    jax.clear_caches()
    perf = am.do_inference_is(y, n_x=1_000)
    print(perf)
    print(am.relmdl.hyl_beliefs)


# Working!!!
def loglin_deg_multilevel():
    boltz_ev = 8.617e-5
    # Model provided in JEDEC's JEP122H as generally used NBTI degradation model, equation 5.3.1,
    # log(1000) term to convert to mV
    def dvth_mv(l_a, eaa, temp, t, n):
        #return jnp.log(1000) + l_a + ((-eaa) / (boltz_ev * temp)) + jnp.log(t ** n)
        return jnp.exp(jnp.log(1000) + l_a + ((-eaa) / (boltz_ev * temp)) + jnp.log(t ** n))

    mb = stratcona.SPMBuilder(mdl_name='deg-general')

    pos_scale_tf = ComposeTransform([SoftplusTransform(), AffineTransform(0, 0.1)])
    mb.add_hyperlatent('l_a_nom', dists.Normal, {'loc': 14.5, 'scale': 0.0001})
    mb.add_hyperlatent('l_a_chp', dists.Normal, {'loc': 1.6, 'scale': 0.0001}, pos_scale_tf)
    mb.add_hyperlatent('l_a_lot', dists.Normal, {'loc': 2.5, 'scale': 0.0001}, pos_scale_tf)
    mb.add_hyperlatent('eaa_nom', dists.Normal, {'loc': 7, 'scale': 0.0001}, pos_scale_tf)

    mb.add_latent('l_a', nom='l_a_nom', chp='l_a_chp', lot='l_a_lot')
    mb.add_latent('eaa', nom='eaa_nom')

    mb.add_intermediate('degn', dvth_mv)
    mb.add_params(n=0.3, degv=2)
    mb.add_observed('deg', dists.Normal, {'loc': 'degn', 'scale': 'degv'}, 1)

    am = stratcona.AnalysisManager(mb.build_model(), rng_seed=833483473)

    # Set up the test and sample observations
    d = stratcona.TestDef('htol', {'htol': {'lot': 3, 'chp': 77}}, {'htol': {'temp': 400, 't': 1000}})
    am.set_test_definition(d)
    k = rand.key(83434)
    y_s = am.relmdl.sample_new(k, d.dims, d.conds, (), am.relmdl.observes)
    y = {'htol': {'deg': y_s['htol_deg']}}
    print(y)

    # Perform inference using custom importance sampling with the v resampling procedure
    am.relmdl.hyl_beliefs = {'l_a_nom': {'loc': 14.5, 'scale': 0.5}, 'eaa_nom': {'loc': 7, 'scale': 0.3},
                             'l_a_chp': {'loc': 2, 'scale': 0.5}, 'l_a_lot': {'loc': 2, 'scale': 0.5}}
    jax.clear_caches()
    perf = am.do_inference_is(y, n_x=5_000)
    print(perf)
    print(am.relmdl.hyl_beliefs)

    # Now compare to HMC
    am.relmdl.hyl_beliefs = {'l_a_nom': {'loc': 14.5, 'scale': 0.5}, 'eaa_nom': {'loc': 7, 'scale': 0.3},
                             'l_a_chp': {'loc': 2, 'scale': 0.5}, 'l_a_lot': {'loc': 2, 'scale': 0.5}}
    jax.clear_caches()
    am.do_inference(y)
    print(am.relmdl.hyl_beliefs)


# Working!!!
def em_multilevel():
    mb = stratcona.SPMBuilder(mdl_name='Black\'s Electromigration')
    boltz_ev = 8.617e-5
    num_devices = 5
    # Wire area in nm^2
    mb.add_params(vth_typ=0.32, i_base=2.8, wire_area=1.024 * 1000 * 1000, k=boltz_ev)

    # Express wire current density as a function of the number of transistors and the voltage applied
    def j_n(n_fins, vdd, vth_typ, i_base):
        return n_fins * i_base * ((vdd - vth_typ) ** 2)

    # The classic model for electromigration failure estimates, DOI: 10.1109/T-ED.1969.16754
    def l_blacks(jn_wire, temp, em_n, em_eaa, wire_area, k):
        return jnp.log(wire_area) - (em_n * jnp.log(jn_wire)) + (em_eaa / (k * temp)) - jnp.log(10 * 3600)

    # Now correspond the degradation mechanisms to the output values
    mb.add_intermediate('jn_wire', j_n)
    mb.add_intermediate('l_fail', l_blacks)
    mb.add_params(lttf_var=0.1)
    mb.add_observed('lttf', dists.Normal, {'loc': 'l_fail', 'scale': 'lttf_var'}, num_devices)

    var_tf = ComposeTransform([SoftplusTransform(), AffineTransform(0, 0.001)])
    mb.add_hyperlatent('n_nom', dists.Normal, {'loc': 13, 'scale': 0.0001}, AffineTransform(0, 0.1))
    mb.add_hyperlatent('n_dev', dists.Normal, {'loc': 16, 'scale': 0.0001}, var_tf)
    mb.add_hyperlatent('n_chp', dists.Normal, {'loc': 18, 'scale': 0.0001}, var_tf)
    mb.add_hyperlatent('eaa_nom', dists.Normal, {'loc': 40, 'scale': 0.0001}, AffineTransform(0, 0.01))
    mb.add_latent('em_n', 'n_nom', 'n_dev', 'n_chp')
    mb.add_latent('em_eaa', 'eaa_nom')

    # Add the chip-level failure time
    def fail_time(lttf):
        return jnp.exp(jnp.min(lttf))
    mb.add_fail_criterion('lifespan', fail_time)

    # Add the total test duration required to get all 5 lines to fail, how to measure time
    def max_l_ttf(lttf):
        return jnp.max(lttf)
    mb.add_fail_criterion('duration', max_l_ttf)

    am = stratcona.AnalysisManager(mb.build_model(), rng_seed=927428374)

    # Set up the test and sample observations
    d = stratcona.TestDef('htol', {'htol': {'lot': 1, 'chp': 5}}, {'htol': {'vdd': 0.95, 'temp': 400, 'n_fins': 240}})
    am.set_test_definition(d)
    k = rand.key(83434)
    y_s = am.relmdl.sample_new(k, d.dims, d.conds, (), am.relmdl.observes)
    y = {'htol': {'lttf': y_s['htol_lttf']}}
    print(y)

    # Perform inference using custom importance sampling with the v resampling procedure
    am.relmdl.hyl_beliefs = {'n_nom': {'loc': 15, 'scale': 2}, 'n_dev': {'loc': 15, 'scale': 3}, 'n_chp': {'loc': 15, 'scale': 3},
                             'eaa_nom': {'loc': 40, 'scale': 1}}
    jax.clear_caches()
    perf = am.do_inference_is(y, n_x=5_000)
    print(perf)
    print(am.relmdl.hyl_beliefs)

    # Now compare to HMC
    am.relmdl.hyl_beliefs = {'n_nom': {'loc': 15, 'scale': 2}, 'n_dev': {'loc': 15, 'scale': 3}, 'n_chp': {'loc': 15, 'scale': 3},
                             'eaa_nom': {'loc': 40, 'scale': 1}}
    jax.clear_caches()
    am.do_inference(y)
    print(am.relmdl.hyl_beliefs)


def demo_custom_inference():
    """
    This demo demos the effectiveness of the custom inference algorithm.
    """

    ####################################################
    # Define an NBTI empirical model within the SPM framework for inference
    ####################################################
    # Model provided in JEDEC's JEP122H as generally used NBTI degradation model, equation 5.3.1
    def bti_vth_shift_empirical(a0, e_aa, temp, vdd, alpha, time, k, n):
        return 1000 * (a0 * 0.001) * jnp.exp((e_aa * 0.01) / (k * temp)) * (vdd ** alpha) * (time ** (n * 0.1))

    mb = stratcona.SPMBuilder(mdl_name='bti-empirical')
    mb.add_params(k=BOLTZ_EV, zero=0.0, meas_var=2, n_nom=2, alpha=3.5, n=2)

    # Initial parameters are simulating some data to then learn
    mb.add_hyperlatent('a0_nom', dists.Normal, {'loc': 5, 'scale': 0.01})
    mb.add_hyperlatent('e_aa_nom', dists.Normal, {'loc': 6, 'scale': 0.01})
    #mb.add_hyperlatent('alpha_nom', dists.Normal, {'loc': 3.5, 'scale': 0.3})
    #mb.add_hyperlatent('n_nom', dists.Normal, {'loc': 2, 'scale': 0.01})

    var_tf = dists.transforms.ComposeTransform([dists.transforms.SoftplusTransform(), dists.transforms.AffineTransform(0, 0.1)])
    mb.add_hyperlatent('a0_dev', dists.Normal, {'loc': 6, 'scale': 0.01}, transform=var_tf)
    #mb.add_hyperlatent('n_dev', dists.Normal, {'loc': 2, 'scale': 0.01}, transform=var_tf)
    #mb.add_hyperlatent('e_aa_dev', dists.Normal, {'loc': 4, 'scale': 3}, transform=var_tf)
    mb.add_hyperlatent('a0_chp', dists.Normal, {'loc': 2, 'scale': 0.01}, transform=var_tf)
    mb.add_hyperlatent('a0_lot', dists.Normal, {'loc': 3, 'scale': 0.01}, transform=var_tf)

    mb.add_latent('a0', nom='a0_nom', dev='a0_dev', chp='a0_chp', lot='a0_lot')
    mb.add_latent('e_aa', nom='e_aa_nom', dev=None, chp=None, lot=None)
    #mb.add_latent('alpha', nom='alpha_nom', dev=None, chp=None, lot=None)
    #mb.add_latent('n', nom='n_nom', dev='n_dev', chp=None, lot=None)

    mb.add_intermediate('dvth', bti_vth_shift_empirical)

    mb.add_observed('dvth_meas', dists.Normal, {'loc': 'dvth', 'scale': 'meas_var'}, 3)

    am = stratcona.AnalysisManager(mb.build_model(), rng_seed=9861823450)

    ####################################################
    # Define the HTOL test that the experimental data was collected from
    ####################################################
    htol_end_test = stratcona.TestDef('htol', {'e': {'lot': 5, 'chp': 5}, 'f': {'lot': 5, 'chp': 5}},
                                              {'e': {'temp': 125 + CELSIUS_TO_KELVIN, 'vdd': 0.88, 'time': 1000},
                                               'f': {'temp': 55 + CELSIUS_TO_KELVIN, 'vdd': 0.88, 'time': 1000}})
    am.set_test_definition(htol_end_test)

    ####################################################
    # Generate the simulated wear-out measurement data
    ####################################################
    k, ks = rand.split(rand.key(19272347))
    sim = am.relmdl.sample(ks, am.test, keep_sites=['e_dvth_meas', 'f_dvth_meas'])
    y = {'e': {'dvth_meas': sim['e_dvth_meas']}, 'f': {'dvth_meas': sim['f_dvth_meas']}}
    print(f"Sim e - mean: {jnp.mean(y['e']['dvth_meas'])}, dev: {jnp.std(y['e']['dvth_meas'])}")
    print(f"Sim f - mean: {jnp.mean(y['f']['dvth_meas'])}, dev: {jnp.std(y['f']['dvth_meas'])}")
    print(f"Model truth - a0_nom: 5, a0_dev: 6, a0_chp: 2, a0_lot: 3, e_aa: 6")

    ####################################################
    # Inference the model using the experimental test data
    ####################################################
    # Set prior beliefs for the model
    #am.relmdl.hyl_beliefs = {'a0_nom': {'loc': 4.0, 'scale': 1.0},
    #                         'a0_dev': {'loc': 7, 'scale': 2},
    #                         'a0_chp': {'loc': 5, 'scale': 2},
    #                         'a0_lot': {'loc': 5, 'scale': 2},
    #                         'e_aa_nom': {'loc': 5, 'scale': 2}}

    #start_time = time.time()
    #am.do_inference(y)
    #print(f'NUTS inference time: {time.time() - start_time}')
    #print(am.relmdl.hyl_beliefs)

    # Reset prior beliefs for the model
    am.relmdl.hyl_beliefs = {'a0_nom': {'loc': 4.0, 'scale': 1.0},
                             'a0_dev': {'loc': 7, 'scale': 2},
                             'a0_chp': {'loc': 5, 'scale': 2},
                             'a0_lot': {'loc': 5, 'scale': 2},
                             'e_aa_nom': {'loc': 5, 'scale': 2}}

    start_time = time.time()
    am.do_inference_mhgibbs(y, n_v=400, beta=0.1)
    print(f'Custom inference time: {time.time() - start_time}')
    print(am.relmdl.hyl_beliefs)


def demo_custom_nom_only():
    """
    This demo demos the effectiveness of the custom inference algorithm.
    """

    ####################################################
    # Define an NBTI empirical model within the SPM framework for inference
    ####################################################
    # Model provided in JEDEC's JEP122H as generally used NBTI degradation model, equation 5.3.1
    def bti_vth_shift_empirical(a0, e_aa, temp, vdd, alpha, time, k, n):
        return 1000 * (a0 * 0.001) * jnp.exp((e_aa * 0.01) / (k * temp)) * (vdd ** alpha) * (time ** (n * 0.1))

    mb = stratcona.SPMBuilder(mdl_name='bti-empirical')
    mb.add_params(k=BOLTZ_EV, zero=0.0, meas_var=1, n_nom=2, alpha=3.5, n=2)

    # Initial parameters are simulating some data to then learn
    mb.add_hyperlatent('a0_nom', dists.Normal, {'loc': 5, 'scale': 0.01})
    mb.add_hyperlatent('e_aa_nom', dists.Normal, {'loc': 6, 'scale': 0.01})
    # mb.add_hyperlatent('alpha_nom', dists.Normal, {'loc': 3.5, 'scale': 0.3})
    # mb.add_hyperlatent('n_nom', dists.Normal, {'loc': 2, 'scale': 0.01})

    mb.add_latent('a0', nom='a0_nom', dev=None, chp=None, lot=None)
    mb.add_latent('e_aa', nom='e_aa_nom', dev=None, chp=None, lot=None)
    # mb.add_latent('alpha', nom='alpha_nom', dev=None, chp=None, lot=None)
    # mb.add_latent('n', nom='n_nom', dev='n_dev', chp=None, lot=None)

    mb.add_intermediate('dvth', bti_vth_shift_empirical)

    mb.add_observed('dvth_meas', dists.Normal, {'loc': 'dvth', 'scale': 'meas_var'}, 5)

    am = stratcona.AnalysisManager(mb.build_model(), rng_seed=9861823450)

    ####################################################
    # Define the HTOL test that the experimental data was collected from
    ####################################################
    htol_end_test = stratcona.ReliabilityTest({'g': {'lot': 1, 'chp': 1}, 'f': {'lot': 1, 'chp': 1}, 'h': {'lot': 1, 'chp': 1}},
                                              {'g': {'temp': 125 + CELSIUS_TO_KELVIN, 'vdd': 0.88, 'time': 1000},
                                               'f': {'temp': 55 + CELSIUS_TO_KELVIN, 'vdd': 0.88, 'time': 1000},
                                               'h': {'temp': 165 + CELSIUS_TO_KELVIN, 'vdd': 0.88, 'time': 1000}})
    am.set_test_definition(htol_end_test)

    ####################################################
    # Generate the simulated wear-out measurement data
    ####################################################
    k, ks = rand.split(rand.key(19272347))
    sim = am.relmdl.sample(ks, am.test, keep_sites=['g_dvth_meas', 'f_dvth_meas', 'h_dvth_meas'])
    y = {'g': {'dvth_meas': sim['g_dvth_meas']}, 'f': {'dvth_meas': sim['f_dvth_meas']}, 'h': {'dvth_meas': sim['h_dvth_meas']}}
    print(f"Sim g - mean: {jnp.mean(y['g']['dvth_meas'])}, dev: {jnp.std(y['g']['dvth_meas'])}")
    print(f"Sim f - mean: {jnp.mean(y['f']['dvth_meas'])}, dev: {jnp.std(y['f']['dvth_meas'])}")
    print(f"Sim h - mean: {jnp.mean(y['h']['dvth_meas'])}, dev: {jnp.std(y['h']['dvth_meas'])}")
    print(f"Model truth - a0_nom: 5, e_aa: 6")

    ####################################################
    # Inference the model using the experimental test data
    ####################################################
    # Set prior beliefs for the model
    #am.relmdl.hyl_beliefs = {'a0_nom': {'loc': 4.0, 'scale': 1.0},
    #                         'e_aa_nom': {'loc': 5, 'scale': 2}}

    #start_time = time.time()
    #am.do_inference(y)
    #print(f'NUTS inference time: {time.time() - start_time}')
    #print(am.relmdl.hyl_beliefs)

    # Reset prior beliefs for the model
    am.relmdl.hyl_beliefs = {'a0_nom': {'loc': 4.0, 'scale': 1.0},
                             'e_aa_nom': {'loc': 5, 'scale': 2}}

    start_time = time.time()
    am.do_inference_mhgibbs(y, num_chains=4, beta=0.1)
    #am.do_inference_custom(y, n_x=10_000)
    print(f'Custom inference time: {time.time() - start_time}')
    print(am.relmdl.hyl_beliefs)


def nbti_nominal_only():
    # Define the simple model
    mb = stratcona.SPMBuilder('idfbcamp-pmos')
    var_tf = dists.transforms.ComposeTransform([dists.transforms.SoftplusTransform(), dists.transforms.AffineTransform(0, 0.1)])

    def nbti_vth(time, vdd, temp, a0, eaa, alpha, n, k):
        return 0.01 * a0 * jnp.exp((eaa * -0.01) / (k * temp)) * (vdd ** alpha) * (time ** (n * 0.1))

    mb.add_hyperlatent('a0n', dists.Normal, {'loc': 3.3, 'scale': 0.0001})
    mb.add_hyperlatent('alphan', dists.Normal, {'loc': 4.0, 'scale': 0.0001})
    mb.add_hyperlatent('nn', dists.Normal, {'loc': 1.9, 'scale': 0.0001})
    mb.add_latent('a0', 'a0n')
    mb.add_latent('alpha', 'alphan')
    mb.add_latent('n', 'nn')
    mb.add_params(time=530, vdd=0.95, temp=400, k=8.617e-5, eaa=5.9)
    mb.add_params(ys=0.04)
    mb.add_intermediate('nbti_vth', nbti_vth)
    mb.add_observed('y', dists.Normal, {'loc': 'nbti_vth', 'scale': 'ys'}, 4)
    am = stratcona.AnalysisManager(mb.build_model(), rng_seed=48)

    # Set up the test and sample observations
    d = stratcona.TestDef('bare', {'e': {'lot': 1, 'chp': 10}}, {'e': {}})
    am.set_test_definition(d)
    k = rand.key(6536)
    y_s = am.relmdl.sample_new(k, d.dims, d.conds, (), am.relmdl.observes)
    y = {'e': {'y': y_s['e_y']}}
    print(f'Mean - {jnp.mean(y_s["e_y"])}, dev - {jnp.std(y_s["e_y"])}')

    # Perform inference using custom importance sampling with the v resampling procedure
    priors = {'a0n': {'loc': 3.6, 'scale': 1.2}, 'alphan': {'loc': 3.5, 'scale': 0.4}, 'nn': {'loc': 2, 'scale': 0.1}}
    am.relmdl.hyl_beliefs = priors
    jax.clear_caches()
    perf = am.do_inference_is(y, n_x=10_000, n_v=500)
    print(perf)
    print(am.relmdl.hyl_beliefs)
    jax.clear_caches()

    # Now compare to HMC
    am.relmdl.hyl_beliefs = priors
    am.do_inference(y)
    print(am.relmdl.hyl_beliefs)


def nbti_single_level_var():
    mb = stratcona.SPMBuilder('idfbcamp-pmos')
    var_tf = dists.transforms.ComposeTransform([dists.transforms.SoftplusTransform(), dists.transforms.AffineTransform(0, 0.1)])

    def nbti_vth(time, vdd, temp, a0, eaa, alpha, n, k):
        return 0.01 * a0 * jnp.exp((eaa * -0.01) / (k * temp)) * (vdd ** alpha) * (time ** (n * 0.1))

    mb.add_hyperlatent('a0n', dists.Normal, {'loc': 3.3, 'scale': 0.0001})
    mb.add_hyperlatent('a0d', dists.Normal, {'loc': 5.6, 'scale': 0.0001}, var_tf)
    mb.add_hyperlatent('alphan', dists.Normal, {'loc': 4.0, 'scale': 0.0001})
    mb.add_hyperlatent('nn', dists.Normal, {'loc': 1.9, 'scale': 0.0001})
    mb.add_latent('a0', 'a0n', 'a0d')
    mb.add_latent('alpha', 'alphan')
    mb.add_latent('n', 'nn')
    mb.add_params(time=530, vdd=0.95, temp=400, k=8.617e-5, eaa=5.9)
    mb.add_params(ys=0.04)
    mb.add_intermediate('nbti_vth', nbti_vth)
    mb.add_observed('y', dists.Normal, {'loc': 'nbti_vth', 'scale': 'ys'}, 4)
    am = stratcona.AnalysisManager(mb.build_model(), rng_seed=48)

    # Set up the test and sample observations
    d = stratcona.TestDef('bare', {'e': {'lot': 1, 'chp': 5}}, {'e': {}})
    am.set_test_definition(d)
    k = rand.key(6536)
    y_s = am.relmdl.sample_new(k, d.dims, d.conds, (), am.relmdl.observes)
    y = {'e': {'y': y_s['e_y']}}
    print(f'Mean - {jnp.mean(y_s["e_y"])}, dev - {jnp.std(y_s["e_y"])}')

    # Perform inference using custom importance sampling with the v resampling procedure
    priors = {'a0n': {'loc': 3.6, 'scale': 1.2}, 'a0d': {'loc': 6.3, 'scale': 3.5},
              'alphan': {'loc': 3.5, 'scale': 0.4}, 'nn': {'loc': 2, 'scale': 0.1}}
    am.relmdl.hyl_beliefs = priors
    jax.clear_caches()
    perf = am.do_inference_is(y, n_x=10_000, n_v=500)
    print(perf)
    print(am.relmdl.hyl_beliefs)
    jax.clear_caches()

    # Now compare to HMC
    am.relmdl.hyl_beliefs = priors
    am.do_inference(y)
    print(am.relmdl.hyl_beliefs)


if __name__ == '__main__':
    em_multilevel()
