# Copyright (c) 2024 Ian Hill
# SPDX-License-Identifier: Apache-2.0

from pprint import pprint
import jax
import jax.numpy as jnp
import jax.random as rand
import numpy as np
import numpyro as npyro
import numpyro.distributions as dists

from functools import partial
from scipy.optimize import curve_fit
from numpy.linalg import cholesky
import reliability
import pandas as pd

import seaborn as sb
from matplotlib import pyplot as plt

import gracefall
import gerabaldi
from gerabaldi.models import *

import os
import sys
# This line adds the parent directory to the module search path so that the Stratcona module can be seen and imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import stratcona

BOLTZ_EV = 8.617e-5
CELSIUS_TO_KELVIN = 273.15


def vth_sensor_inference():
    npyro.set_host_device_count(4)

    # This example must show the benefits of Bayesian inference in reasoning about physical wear-out models, demo how
    # historical knowledge incorporation, explicit uncertainty reasoning, stochastic variability modelling, and
    # appropriate reasoning with limited data all
    # allow for a more realistic and effective assessment of reliability. Compare to point estimate + covariance fitting
    # via regression models, try and follow some frequentist paper.

    # Helper function to convert likelihoods to alpha values that result in clear plots
    def likelihood_to_alpha(probs, max_alpha=0.5):
        min_alpha = 0.1
        alphas = min_alpha + ((probs / jnp.max(probs)) * (max_alpha - min_alpha))
        return alphas

    def avg_same_time_measurements(measd, average=True, normalize=False):
        avgd = pd.DataFrame()
        t_prev = -10000
        # Average the measurements of the same parameter taken near the same time relative to the total test length
        for prm in measd['param'].unique():
            prm_meas = measd.loc[measd['param'] == prm]
            # Averaging for repeated measurement of devices during each measurement time step
            for lot in range(prm_meas['lot #'].min(), prm_meas['lot #'].max() + 1):
                for chp in range(prm_meas['chip #'].min(), prm_meas['chip #'].max() + 1):
                    for dev in range(prm_meas['device #'].min(), prm_meas['device #'].max() + 1):
                        # For every individual device measured, we find measured times that are close together (+-6 mins)
                        dev_meas = prm_meas.loc[(prm_meas['device #'] == dev) &
                                                (prm_meas['chip #'] == chp) &
                                                (prm_meas['lot #'] == lot)]
                        for t in dev_meas['time']:
                            # Only process each time 'group' once
                            if t <= t_prev - 360 or t >= t_prev + 360:
                                meas_group = dev_meas.loc[(dev_meas['time'] >= t - 360) & (dev_meas['time'] <= t + 360)]
                                # Find the average of all the measurements taken near the specified time
                                avg = meas_group['measured'].mean()
                                dev_meas.loc[dev_meas['time'] == t, 'measured'] = avg
                                # Assemble the new dataframe by appending each averaged row in turn
                                avgd = pd.concat((avgd, dev_meas.loc[dev_meas['time'] == t]), ignore_index=True)
                                t_prev = t
                        # print(f"Finished a device for param {prm}!")
        return avgd

    def diff_from_max_val(measd, return_maxes=False):
        vals = measd.copy()
        for prm in vals['param'].unique():
            prm_data = vals.loc[vals['param'] == prm]
            for lot in range(prm_data['lot #'].min(), prm_data['lot #'].max() + 1):
                for chp in range(prm_data['chip #'].min(), prm_data['chip #'].max() + 1):
                    for dev in range(prm_data['device #'].min(), prm_data['device #'].max() + 1):
                        max_val = prm_data.loc[(prm_data['device #'] == dev) & (prm_data['chip #'] == chp) & (
                                    prm_data['lot #'] == lot), 'measured'].max()
                        if return_maxes:
                            vals.loc[(vals['param'] == prm) & (vals['device #'] == dev) & (vals['chip #'] == chp) & (
                                        vals['lot #'] == lot), 'measured'] = max_val
                        else:
                            vals.loc[(vals['param'] == prm) & (vals['device #'] == dev) & (vals['chip #'] == chp) & (
                                        vals['lot #'] == lot), 'measured'] -= max_val

        return vals

    def diff_from_init_val(measd, return_inits=False):
        vals = measd.copy()
        for prm in vals['param'].unique():
            prm_data = vals.loc[vals['param'] == prm]
            for lot in range(prm_data['lot #'].min(), prm_data['lot #'].max() + 1):
                for chp in range(prm_data['chip #'].min(), prm_data['chip #'].max() + 1):
                    for dev in range(prm_data['device #'].min(), prm_data['device #'].max() + 1):
                        init_val = prm_data.loc[
                            (prm_data['device #'] == dev) & (prm_data['chip #'] == chp) & (prm_data['lot #'] == lot) & (
                                        prm_data['time'] == 0), 'measured'].max()
                        if return_inits:
                            vals.loc[(vals['param'] == prm) & (vals['device #'] == dev) & (vals['chip #'] == chp) & (
                                        vals['lot #'] == lot), 'measured'] = init_val
                        else:
                            vals.loc[(vals['param'] == prm) & (vals['device #'] == dev) & (vals['chip #'] == chp) & (
                                        vals['lot #'] == lot), 'measured'] -= init_val

        return vals

    def rename_series(measd, prm, new_name, device):
        new_df = measd.copy()
        new_df.loc[new_df['param'] == prm, 'device #'] = device
        new_df.loc[new_df['param'] == prm, 'param'] = new_name
        return new_df

    # Import the data from a JSON file
    h_data = gracefall.load_gerabaldi_report('C:/Users/Ian Hill/PycharmProjects/hill-scripts/data_files/idfbcamp_htol_lt_meas.json')
    h_data = h_data['Measurements']
    # Remove the measurement time that was only conducted for one of the lots
    h_data.drop(h_data[h_data['time'] == 3034800].index, inplace=True)
    h_data.drop(h_data[h_data['time'] == 3034860].index, inplace=True)
    h_data.drop(h_data[h_data['time'] == 3034920].index, inplace=True)
    # Rename direct measurement sensors
    h_data = rename_series(h_data, prm='pbti_vth_direct', new_name='pbti_vth', device=4)
    h_data = rename_series(h_data, prm='nbti_vth_direct', new_name='nbti_vth', device=4)
    h_data = rename_series(h_data, prm='nhci_vth_direct', new_name='nhci_vth', device=4)
    h_data = rename_series(h_data, prm='phci_vth_direct', new_name='phci_vth', device=4)
    # Drop all but the vth sensor measurements
    sensor_types = ['pbti_vth', 'nbti_vth', 'nhci_vth', 'phci_vth']
    h_data = h_data[h_data['param'].isin(sensor_types)]
    # Average the three measurements taken at each 100 hour mark
    h_data = avg_same_time_measurements(h_data)
    # Now extract the vth shift as the difference from the max value
    h_data = diff_from_init_val(h_data)
    # Don't care about which board the chip was on, just the chip IDs
    h_data['chip #'] = h_data['chip #'] + (2 * h_data['lot #'])

    deg_data = {}
    for t in jnp.linspace(0, 3_600_000, 11):
        hours = int(t / 3600)
        deg_data[f't{hours}'] = {}
        for sensor in sensor_types:
            vth = h_data.loc[(h_data['param'] == sensor) & (h_data['time'] == t)].drop(columns=['param', 'lot #', 'time'])
            vth = vth.sort_values(['chip #', 'device #'], ascending=[True, True])
            vth_array = jnp.array(vth['measured'].values).reshape((vth['chip #'].nunique(), vth['device #'].nunique()))
            deg_data[f't{hours}'][sensor] = vth_array * 1000

    ####################################################
    # Define the BTI SPM for inference
    ####################################################
    # Model provided in JEDEC's JEP122H as generally used NBTI degradation model, equation 5.3.1
    def bti_vth_shift_empirical(a0, e_aa, temp, vdd, alpha, time, k, n):
        return 1000 * (a0 * 0.001) * jnp.exp((e_aa * 0.01) / (k * temp)) * (vdd ** alpha) * (time ** (n * 0.1))
    def bti_vth_shift_empirical_t2(a0, e_aa, temp, vdd, alpha, time2, k, n):
        return 1000 * (a0 * 0.001) * jnp.exp((e_aa * 0.01) / (k * temp)) * (vdd ** alpha) * (time2 ** (n * 0.1))
    def bti_vth_shift_empirical_t3(a0, e_aa, temp, vdd, alpha, time3, k, n):
        return 1000 * (a0 * 0.001) * jnp.exp((e_aa * 0.01) / (k * temp)) * (vdd ** alpha) * (time3 ** (n * 0.1))

    mb = stratcona.SPMBuilder(mdl_name='bti-empirical')
    mb.add_params(k=BOLTZ_EV, zero=0.0, meas_var=20)

    mb.add_hyperlatent('a0_nom', dists.Normal, {'loc': 5, 'scale': 2})
    mb.add_hyperlatent('e_aa_nom', dists.Normal, {'loc': 6, 'scale': 2})
    mb.add_hyperlatent('alpha_nom', dists.Normal, {'loc': 3.5, 'scale': 0.3})
    mb.add_hyperlatent('n_nom', dists.Normal, {'loc': 2, 'scale': 0.3})

    mb.add_latent('a0', nom='a0_nom', dev=None, chp=None, lot=None)
    mb.add_latent('e_aa', nom='e_aa_nom', dev=None, chp=None, lot=None)
    mb.add_latent('alpha', nom='alpha_nom', dev=None, chp=None, lot=None)
    mb.add_latent('n', nom='n_nom', dev=None, chp=None, lot=None)

    mb.add_hyperlatent('meas_error_var', dists.Normal, {'loc': 30, 'scale': 10}, transform=dists.transforms.SoftplusTransform())
    mb.add_latent('meas_error', nom='zero', dev='meas_error_var', chp=None, lot=None)

    mb.add_dependent('vth_shift_t1', bti_vth_shift_empirical)
    #mb.add_dependent('vth_shift_t2', bti_vth_shift_empirical_t2)
    #mb.add_dependent('vth_shift_t3', bti_vth_shift_empirical_t3)

    # FIXME: Currently using a fixed measurement variability of 30mV since the latent version is causing NUTS to fail,
    #        determine the cause and evaluate the best path forward
    mb.add_measured('nbti_vth', dists.Normal, {'loc': 'vth_shift_t1', 'scale': 'meas_var'}, 5)
    #mb.add_measured('nbti_vth_t2', dists.Normal, {'loc': 'vth_shift_t2', 'scale': 'meas_var'}, 5)
    #mb.add_measured('nbti_vth_t3', dists.Normal, {'loc': 'vth_shift_t3', 'scale': 'meas_var'}, 5)

    am = stratcona.AnalysisManager(mb.build_model(), rng_seed=9861823450)

    # TODO: Add experiment steps to SPM definitions, allowing for experiments with multiple measurements of
    #       the same device
    htol_end_test = stratcona.ReliabilityTest({'t1000': {'lot': 1, 'chp': 4}, 't800': {'lot': 1, 'chp': 4}, 't600': {'lot': 1, 'chp': 4}},
                                              {'t1000': {'temp': 125 + CELSIUS_TO_KELVIN, 'vdd': 0.88, 'time': 1000},
                                               't800': {'temp': 125 + CELSIUS_TO_KELVIN, 'vdd': 0.88, 'time': 800},
                                               't600': {'temp': 125 + CELSIUS_TO_KELVIN, 'vdd': 0.88, 'time': 600}})
    am.set_test_definition(htol_end_test)

    # Sample some projected degradation measurements from prior beliefs for calibration
    sample_deg = am.sim_test_measurements()

    k1, k2 = rand.split(rand.key(292873410), 2)
    prm_samples = am.relmdl.sample(k1, htol_end_test, 400)
    ltnt_sites = ['a0_nom', 'e_aa_nom', 'alpha_nom', 'n_nom']
    ltnt_vals = {site: data for site, data in prm_samples.items() if site in ltnt_sites}
    pri_probs = jnp.exp(am.relmdl.logp(k2, htol_end_test, ltnt_vals, prm_samples))
    pri_probs = likelihood_to_alpha(pri_probs, 0.4).flatten()

    times = jnp.tile(jnp.linspace(0, 2000, 100), (400, 1)).T
    pri_fits = bti_vth_shift_empirical(
        time=times, k=BOLTZ_EV, temp=125 + CELSIUS_TO_KELVIN, vdd=0.88,
        a0=prm_samples['a0_nom'], e_aa=prm_samples['e_aa_nom'],
        alpha=prm_samples['alpha_nom'], n=prm_samples['n_nom'])

    # Generate hyperlatent prior data for plotting and compute entropy
    def lp_f(vals, site, key, test, trace):
        return am.relmdl.logp(rng_key=key, test=test, site_vals={site: vals}, conditional=trace)

    k1, k2 = rand.split(rand.key(9273036857), 2)
    hyl_samples = am.relmdl.sample(k1, htol_end_test, 100_000)
    hyls = ['a0', 'e_aa', 'alpha', 'n']
    tr = am.relmdl.sample(k2, htol_end_test, 1)
    pri_samples, pri_entropy = {}, {}
    for hyl in hyls:
        pri_samples[hyl] = hyl_samples[f'{hyl}_nom']
        pri_entropy[hyl] = stratcona.engine.boed.entropy_new(
            pri_samples[hyl], partial(lp_f, site=f'{hyl}_nom', test=htol_end_test, trace=tr, key=k1), limiting_density_range=(0, 10))

    ####################################################
    # Inference on the collected test data
    ####################################################
    am.do_inference(deg_data)

    # Calculate posterior entropy
    k1, k2 = rand.split(rand.key(9296245908724), 2)
    hyl_samples = am.relmdl.sample(k1, htol_end_test, 100_000)
    hyls = ['a0', 'e_aa', 'alpha', 'n']
    tr = am.relmdl.sample(k2, htol_end_test, 1)
    pst_samples, pst_entropy = {}, {}
    for hyl in hyls:
        pst_samples[hyl] = hyl_samples[f'{hyl}_nom']
        pst_entropy[hyl] = stratcona.engine.boed.entropy_new(
            pst_samples[hyl], partial(lp_f, site=f'{hyl}_nom', test=htol_end_test, trace=tr, key=k1), limiting_density_range=(0, 10))

    hyl_ig = {}
    for hyl in hyls:
        hyl_ig[hyl] = pri_entropy[hyl] - pst_entropy[hyl]

    sample_pst_deg = am.sim_test_measurements()

    prm_samples = am.relmdl.sample(k1, htol_end_test, 400)
    ltnt_vals = {site: data for site, data in prm_samples.items() if site in ltnt_sites}
    pst_probs = jnp.exp(am.relmdl.logp(k2, htol_end_test, ltnt_vals, prm_samples))
    pst_probs = likelihood_to_alpha(pst_probs, 0.4).flatten()
    pst_fits = bti_vth_shift_empirical(
        time=times, k=BOLTZ_EV, temp=125 + CELSIUS_TO_KELVIN, vdd=0.88,
        a0=prm_samples['a0_nom'], e_aa=prm_samples['e_aa_nom'],
        alpha=prm_samples['alpha_nom'], n=prm_samples['n_nom'])

    ####################################################
    # Plot the entropy of the hyperlatent parameters
    ####################################################
    sb.set_context('notebook')
    sb.set_theme(style='ticks', font='Times New Roman')

    #fig, p = plt.subplots(2, 2)
    #for i, hyl in enumerate(hyls):
    #    ind0, ind1 = int(i / 2), i % 2
    #    p[ind0][ind1].hist(pri_samples[hyl], 100, color='skyblue', density=True, histtype='stepfilled',
    #                       label='Prior')
    #    p[ind0][ind1].hist(pst_samples[hyl], 100, color='darkblue', alpha=0.5, density=True, histtype='stepfilled',
    #                       label='Posterior')
    #    p[ind0][ind1].text(0.65, 0.65, f'Information gain:\n{round(float(hyl_ig[hyl]), 3)} nats',
    #                       transform=p[ind0][ind1].transAxes)
    #    p[ind0][ind1].set_xlim(0, 10)
    #    p[ind0][ind1].set_ylim(0, 1.5)
    #    leg = p[ind0][ind1].legend(loc='upper left')
    #    for lbl in leg.legend_handles:
    #        lbl.set_alpha(1)
    #    p[ind0][ind1].set_xlabel(hyl, fontsize='medium')
    #    p[ind0][ind1].set_ylabel('PDF', fontsize='medium')

    #fig.subplots_adjust(wspace=0, hspace=0)

    fig, p = plt.subplots(1, 1)
    colours = {'a0': ['skyblue', 'darkblue'], 'e_aa': ['chartreuse', 'green'],
               'alpha': ['orchid', 'darkorchid'], 'n': ['palevioletred', 'crimson']}
    display_map = {'a0': "$A_0$", 'e_aa': "$E_{aa}$", 'alpha': "$\\alpha$", 'n': "$n$"}
    hyls_ordered = ['n', 'alpha', 'a0', 'e_aa']
    for hyl in hyls_ordered:
        p.hist(pri_samples[hyl], 100, color=colours[hyl][0], alpha=0.8, density=True, histtype='stepfilled',
                           label=f'{display_map[hyl]} prior')
        p.hist(pst_samples[hyl], 100, color=colours[hyl][1], alpha=0.5, density=True, histtype='stepfilled',
                           label=f'{display_map[hyl]} posterior - IG: {round(float(hyl_ig[hyl]), 3)} nats')
    p.set_xlim(0, 10)
    p.set_ylim(0, 1.5)
    leg = p.legend(loc='upper right')
    for lbl in leg.legend_handles:
        lbl.set_alpha(1)
    p.set_xlabel('Parameter value', fontsize='medium')
    p.set_ylabel('PDF', fontsize='medium')

    ####################################################
    # Plot the prior and posterior predictive curves
    ####################################################
    #fig, p = plt.subplots(1, 1)
    #p.grid()
    #sb.stripplot({'pri': sample_deg['t1000']['pbti_vth'].flatten(),
    #              'pst': sample_pst_deg['t1000']['pbti_vth'].flatten()}, ax=p)

    fig, p = plt.subplots(1, 1)
    # Plot the probabilistic fits
    times, pri_fits, pst_fits = times.T, pri_fits.T, pst_fits.T
    for i in range(len(pri_fits)):
        lbl = 'Prior predictive distribution' if i == 0 else None
        p.plot(times[i], pri_fits[i].flatten(), alpha=float(pri_probs[i]), color='skyblue', label=lbl)
    for i in range(len(pst_fits)):
        lbl = 'Posterior predictive distribution' if i == 0 else None
        p.plot(times[i], pst_fits[i].flatten(), alpha=float(pst_probs[i]), color='darkblue', label=lbl)

    # Add the measured data that was used for inference
    measd_vals = deg_data['t1000']['nbti_vth'].flatten()
    p.plot(jnp.full((len(measd_vals),), 1000), measd_vals, color='darkorange', linestyle='', marker='.', markersize=8,
           label='Observed degradation')
    measd_vals = deg_data['t800']['nbti_vth'].flatten()
    p.plot(jnp.full((len(measd_vals),), 800), measd_vals, color='darkorange', linestyle='', marker='.', markersize=8)
    measd_vals = deg_data['t600']['nbti_vth'].flatten()
    p.plot(jnp.full((len(measd_vals),), 600), measd_vals, color='darkorange', linestyle='', marker='.', markersize=8)

    leg = p.legend(loc='upper right')
    for lbl in leg.legend_handles:
        lbl.set_alpha(1)
    p.set_xlim(0, 1100)
    p.set_xlabel('Time (hours)', fontsize='medium')
    p.set_ylim(-50, 200)
    p.set_ylabel("$\Delta V_{th}$ (mV)", fontsize='medium')

    plt.show()






if __name__ == '__main__':
    vth_sensor_inference()
