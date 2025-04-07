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

import seaborn as sb
from matplotlib import pyplot as plt
import matplotlib.lines as pltlines

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


# Helper function to convert likelihoods to RGBA alpha values that result in clear plots
def likelihood_to_alpha(probs, max_alpha=0.5):
    min_alpha = 0.1
    alphas = min_alpha + ((probs / jnp.max(probs)) * (max_alpha - min_alpha))
    return alphas


def vth_sensor_inference():
    """
    This study shows the application of Bayesian inference to an NBTI physical model based on experimental data. The
    example avoids overfit to a very small dataset, shows chip-level variability modelling, and visualization of LDDP
    entropy for epistemic uncertainty quantification.
    """

    ####################################################
    # First load in the experimental data used for inference.
    ####################################################
    with open('./bti_ro_data.json', 'r') as f:
        deg_data = json.load(f)
    deg_data['t1000']['nbti_std_ro'] = jnp.array(deg_data['t1000']['nbti_std_ro'])

    ####################################################
    # Define the NBTI empirical model within the SPM framework for inference
    ####################################################
    # Model provided in JEDEC's JEP122H as generally used NBTI degradation model, equation 5.3.1
    def bti_vth_shift_empirical(a0, e_aa, temp, vdd, alpha, time, k, n):
        return 1000 * (a0 * 0.001) * jnp.exp((e_aa * 0.01) / (k * temp)) * (vdd ** alpha) * (time ** (n * 0.1))

    mb = stratcona.SPMBuilder(mdl_name='bti-empirical')
    mb.add_params(k=BOLTZ_EV, zero=0.0, meas_var=5)

    mb.add_hyperlatent('a0_nom', dists.Normal, {'loc': 5, 'scale': 2})
    mb.add_hyperlatent('e_aa_nom', dists.Normal, {'loc': 6, 'scale': 2})
    mb.add_hyperlatent('alpha_nom', dists.Normal, {'loc': 3.5, 'scale': 0.3})
    mb.add_hyperlatent('n_nom', dists.Normal, {'loc': 2, 'scale': 0.3})

    var_tf = dists.transforms.ComposeTransform([dists.transforms.SoftplusTransform(), dists.transforms.AffineTransform(0, 0.1)])
    mb.add_hyperlatent('a0_dev', dists.Normal, {'loc': 6, 'scale': 4}, transform=var_tf)
    mb.add_hyperlatent('e_aa_dev', dists.Normal, {'loc': 6, 'scale': 3}, transform=var_tf)
    mb.add_hyperlatent('a0_chp', dists.Normal, {'loc': 6, 'scale': 4}, transform=var_tf)

    mb.add_latent('a0', nom='a0_nom', dev='a0_dev', chp='a0_chp', lot=None)
    mb.add_latent('e_aa', nom='e_aa_nom', dev='e_aa_dev', chp=None, lot=None)
    mb.add_latent('alpha', nom='alpha_nom', dev=None, chp=None, lot=None)
    mb.add_latent('n', nom='n_nom', dev=None, chp=None, lot=None)

    mb.add_intermediate('vth_shift_t1', bti_vth_shift_empirical)

    mb.add_observed('nbti_std_ro', dists.Normal, {'loc': 'vth_shift_t1', 'scale': 'meas_var'}, 4)

    am = stratcona.AnalysisManager(mb.build_model(), rng_seed=9861823450)


    ####################################################
    # Define the HTOL test that the experimental data was collected from
    ####################################################
    htol_end_test = stratcona.TestDef('htol_end', {'t1000': {'lot': 1, 'chp': 4}},
                                      {'t1000': {'temp': 125 + CELSIUS_TO_KELVIN, 'vdd': 0.88, 'time': 1000}})
    am.set_test_definition(htol_end_test)

    ######################################################
    # Render the SPM for viewing
    ######################################################
    dims = htol_end_test.dims
    conds = htol_end_test.conds
    priors = am.relmdl.hyl_beliefs
    params = am.relmdl.param_vals
    npyro.render_model(am.relmdl.spm, model_args=(dims, conds, priors, params),
                       filename='renders/nbti_model.png')

    ####################################################
    # Generate some prior predictive curves and their epistemic probabilities
    ####################################################
    k1, k2 = rand.split(rand.key(292873410), 2)
    prm_samples = am.relmdl.sample(k1, htol_end_test, (400,))
    ltnt_sites = ['a0_nom', 'e_aa_nom', 'alpha_nom', 'n_nom']
    ltnt_vals = {site: data for site, data in prm_samples.items() if site in ltnt_sites}
    pri_probs = jnp.exp(am.relmdl.logp(k2, htol_end_test, ltnt_vals, prm_samples, (400,)))
    pri_probs = likelihood_to_alpha(pri_probs, 0.4).flatten()

    times = jnp.tile(jnp.linspace(0, 2000, 100), (400, 1)).T
    pri_fits = bti_vth_shift_empirical(
        time=times, k=BOLTZ_EV, temp=125 + CELSIUS_TO_KELVIN, vdd=0.88,
        a0=prm_samples['a0_nom'], e_aa=prm_samples['e_aa_nom'],
        alpha=prm_samples['alpha_nom'], n=prm_samples['n_nom'])

    ####################################################
    # Generate samples of the prior hyper-latents for computing and plotting LDDP entropy
    ####################################################
    def lp_f(vals, site, key, test):
        return am.relmdl.logp(rng_key=key, test=test, site_vals={site: vals}, conditional=None, dims=(len(vals),))

    k1, k2 = rand.split(rand.key(9273036857), 2)
    hyl_samples = am.relmdl.sample(k1, htol_end_test, (ENTROPY_SAMPLES,))
    hyls = ['a0_nom', 'e_aa_nom', 'alpha_nom', 'n_nom', 'a0_dev', 'e_aa_dev', 'a0_chp']
    pri_samples, pri_entropy = {}, {}
    for hyl in hyls:
        pri_samples[hyl] = hyl_samples[hyl]
        pri_entropy[hyl] = stratcona.engine.bed.entropy(
            pri_samples[hyl], partial(lp_f, site=hyl, test=htol_end_test, key=k1), limiting_density_range=(-400, 400))

    ####################################################
    # Inference the model using the experimental test data
    ####################################################
    start_time = time.time()
    inf_data = {'t1000': {'nbti_std_ro': deg_data['t1000']['nbti_std_ro']}}
    am.do_inference(inf_data)
    print(f'Inference time taken: {time.time() - start_time}')
    print(am.relmdl.hyl_beliefs)

    ####################################################
    # Generate samples of the posterior hyper-latents for computing and plotting LDDP entropy
    ####################################################
    # Calculate posterior entropy
    k1, k2 = rand.split(rand.key(9296245908724), 2)
    hyl_samples = am.relmdl.sample(k1, htol_end_test, (ENTROPY_SAMPLES,))
    pst_samples, pst_entropy = {}, {}
    for hyl in hyls:
        pst_samples[hyl] = hyl_samples[hyl]
        pst_entropy[hyl] = stratcona.engine.bed.entropy(
            pst_samples[hyl], partial(lp_f, site=hyl, test=htol_end_test, key=k1), limiting_density_range=(-400, 400))

    hyl_ig = {}
    for hyl in hyls:
        hyl_ig[hyl] = pri_entropy[hyl] - pst_entropy[hyl]

    ####################################################
    # Generate some posterior predictive curves and their epistemic probabilities
    ####################################################
    prm_samples = am.relmdl.sample(k1, htol_end_test, (400,))
    ltnt_vals = {site: data for site, data in prm_samples.items() if site in ltnt_sites}
    pst_probs = jnp.exp(am.relmdl.logp(k2, htol_end_test, ltnt_vals, prm_samples, (400,)))
    pst_probs = likelihood_to_alpha(pst_probs, 0.4).flatten()
    pst_fits = bti_vth_shift_empirical(
        time=times, k=BOLTZ_EV, temp=125 + CELSIUS_TO_KELVIN, vdd=0.88,
        a0=prm_samples['a0_nom'], e_aa=prm_samples['e_aa_nom'],
        alpha=prm_samples['alpha_nom'], n=prm_samples['n_nom'])

    ####################################################
    # Plot the entropy of the hyper-latent parameters
    ####################################################
    sb.set_context('notebook')
    sb.set_theme(style='ticks', font='Times New Roman')

    fig, p = plt.subplots(1, 1)
    display_map = {'a0_nom': "$\\mu_{A_0}$", 'e_aa_nom': "$\\mu_{E_{aa}}$", 'alpha_nom': "$\\alpha$", 'n_nom': "$\\mu_n$",
                   'a0_dev': "$\\sigma_{A_0}$", 'e_aa_dev': "$\\sigma_{E_{aa}}$", 'alpha_dev': "$\\sigma_{\\alpha}$", 'n_dev': "$\\sigma_{n}$",
                   'a0_chp': "$\\varsigma_{A_0}$"}
    df_list = []
    hyl_subset = ['a0_nom', 'a0_dev', 'a0_chp', 'e_aa_nom', 'e_aa_dev', 'n_nom']
    for hyl in hyl_subset:
        hyl_df = pd.DataFrame(pri_samples[hyl], columns=['val'])
        hyl_df['hyl'] = display_map[hyl]
        hyl_df['pri-pst'] = 'pri'
        df_list.append(hyl_df)
    for hyl in hyl_subset:
        hyl_df = pd.DataFrame(pst_samples[hyl], columns=['val'])
        hyl_df['hyl'] = display_map[hyl]
        hyl_df['pri-pst'] = 'pst'
        df_list.append(hyl_df)
    df_violin = pd.concat(df_list)

    sb.violinplot(df_violin, x='val', y='hyl', ax=p, split=True, density_norm='count',
                  hue='pri-pst', inner='quart', palette=['skyblue', 'darkblue'], linewidth=1.25)
    for fill in p.collections:
        fill.set_alpha(0.75)

    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Times New Roman'
    plt.rcParams['mathtext.it'] = 'Times New Roman'
    plt.rcParams['font.family'] = 'Times New Roman'
    # Add text annotations showing LDDP entropy
    pos = range(len(hyls))
    for tick, label, hyl in zip(pos, p.get_yticklabels(), hyl_subset):
        p.text(-2, pos[tick] - 0.11,
               f"$H_{{prior}}={round(float(pri_entropy[hyl]), 2)}$",
               horizontalalignment='center', size='medium', color='black')
        p.text(-2, pos[tick] + 0.26,
               f'$H_{{posterior}}={round(float(pst_entropy[hyl]), 2)}$',
               horizontalalignment='center', size='medium', color='black')
        p.text(13, pos[tick] + 0.26,
               f'$IG={round(float(hyl_ig[hyl]), 2)} \\;\\; nats$',
               horizontalalignment='center', size='medium', color='black')
    p.legend().remove()
    p.tick_params(axis='y', which='major', labelsize=13, labelfontfamily='Times New Roman')
    p.set_xlabel('Value distribution', fontsize='medium')
    p.set_ylabel('Hyper-latent variable', fontsize='medium')

    ####################################################
    # Plot the prior and posterior predictive curves
    ####################################################
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
    colours = ['darkorange', 'sienna', 'gold', 'firebrick']
    for chp, clr in enumerate(colours):
        measd_vals = deg_data['t1000']['nbti_std_ro'][chp]
        p.plot(jnp.full((len(measd_vals),), 1000), measd_vals, color=clr, linestyle='', marker='.',
               markersize=8, label=None)

    hndl, lbls = p.get_legend_handles_labels()
    lgnd2 = pltlines.Line2D([0], [0], color='black', linestyle='', marker='.')
    hndl.insert(0, lgnd2)
    lbls.insert(0, 'Simulated data, coloured by chip')

    leg = p.legend(hndl, lbls, loc='lower right')
    for lbl in leg.legend_handles:
        lbl.set_alpha(1)
    p.set_xlim(0, 1100)
    p.set_yticks([])
    p.set_xlabel('Time (hours)', fontsize='medium')
    p.set_ylim(-50, 200)
    p.set_ylabel("$\Delta V_{th}$ (A.U.)", fontsize='medium')

    plt.show()


if __name__ == '__main__':
    vth_sensor_inference()
