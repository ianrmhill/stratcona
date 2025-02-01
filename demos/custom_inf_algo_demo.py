# Copyright (c) 2025 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import numpyro as npyro
import numpyro.distributions as dists
# Device count has to be set before importing jax
npyro.set_host_device_count(4)

import jax.numpy as jnp
import jax.random as rand

import time
from functools import partial
import json
import pandas as pd
from io import StringIO

import seaborn as sb
from matplotlib import pyplot as plt

import gerabaldi
from gerabaldi.models import *

import os
import sys
# This line adds the parent directory to the module search path so that the Stratcona module can be seen and imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import stratcona

BOLTZ_EV = 8.617e-5
CELSIUS_TO_KELVIN = 273.15
ENTROPY_SAMPLES = 100_000


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
    mb.add_params(k=BOLTZ_EV, zero=0.0, meas_var=2, n_nom=2, e_aa=5, alpha=3.5, n=2)

    # Initial parameters are simulating some data to then learn
    mb.add_hyperlatent('a0_nom', dists.Normal, {'loc': 5, 'scale': 0.01})
    #mb.add_hyperlatent('e_aa_nom', dists.Normal, {'loc': 6, 'scale': 2})
    #mb.add_hyperlatent('alpha_nom', dists.Normal, {'loc': 3.5, 'scale': 0.3})
    #mb.add_hyperlatent('n_nom', dists.Normal, {'loc': 2, 'scale': 0.01})

    var_tf = dists.transforms.ComposeTransform([dists.transforms.SoftplusTransform(), dists.transforms.AffineTransform(0, 0.1)])
    mb.add_hyperlatent('a0_dev', dists.Normal, {'loc': 6, 'scale': 0.01}, transform=var_tf)
    #mb.add_hyperlatent('n_dev', dists.Normal, {'loc': 2, 'scale': 0.01}, transform=var_tf)
    #mb.add_hyperlatent('e_aa_dev', dists.Normal, {'loc': 4, 'scale': 3}, transform=var_tf)
    mb.add_hyperlatent('a0_chp', dists.Normal, {'loc': 3, 'scale': 0.01}, transform=var_tf)

    mb.add_latent('a0', nom='a0_nom', dev='a0_dev', chp='a0_chp', lot=None)
    #mb.add_latent('e_aa', nom='e_aa_nom', dev='e_aa_dev', chp=None, lot=None)
    #mb.add_latent('alpha', nom='alpha_nom', dev=None, chp=None, lot=None)
    #mb.add_latent('n', nom='n_nom', dev='n_dev', chp=None, lot=None)

    mb.add_intermediate('dvth', bti_vth_shift_empirical)

    mb.add_observed('dvth_meas', dists.Normal, {'loc': 'dvth', 'scale': 'meas_var'}, 10)

    am = stratcona.AnalysisManager(mb.build_model(), rng_seed=9861823450)

    ####################################################
    # Define the HTOL test that the experimental data was collected from
    ####################################################
    htol_end_test = stratcona.ReliabilityTest({'e': {'lot': 1, 'chp': 10}},
                                              {'e': {'temp': 125 + CELSIUS_TO_KELVIN, 'vdd': 0.88, 'time': 1000}})
    am.set_test_definition(htol_end_test)

    ####################################################
    # Generate the simulated wear-out measurement data
    ####################################################
    k, ks = rand.split(rand.key(19272347))
    y = {'e': {'dvth_meas': am.relmdl.sample(ks, am.test, keep_sites=['e_dvth_meas'])['e_dvth_meas']}}
    print(f"Sim - mean: {jnp.mean(y['e']['dvth_meas'])}, dev: {jnp.std(y['e']['dvth_meas'])}")
    print(f"Model truth - a0_nom: 5, a0_dev: 6, a0_chp: 3")

    ####################################################
    # Inference the model using the experimental test data
    ####################################################
    # Set prior beliefs for the model
    am.relmdl.hyl_beliefs = {'a0_nom': {'loc': 4.0, 'scale': 1.0},
                             'a0_dev': {'loc': 7, 'scale': 2},
                             'a0_chp': {'loc': 5, 'scale': 2}}

    start_time = time.time()
    am.do_inference(y)
    print(f'NUTS inference time: {time.time() - start_time}')
    print(am.relmdl.hyl_beliefs)

    # Reset prior beliefs for the model
    am.relmdl.hyl_beliefs = {'a0_nom': {'loc': 4.0, 'scale': 1.0},
                             'a0_dev': {'loc': 7, 'scale': 2},
                             'a0_chp': {'loc': 5, 'scale': 2}}

    start_time = time.time()
    am.do_inference_custom(y)
    print(f'Custom inference time: {time.time() - start_time}')
    print(am.relmdl.hyl_beliefs)


if __name__ == '__main__':
    demo_custom_inference()
