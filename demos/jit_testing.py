
import numpyro as npyro
import numpyro.distributions as dists
from numpyro.handlers import seed, trace, condition

npyro.set_host_device_count(4)

import jax.numpy as jnp
import jax.random as rand
import jax

from dataclasses import dataclass
from functools import partial
import timeit
import time
import json
import struct

import seaborn
import matplotlib.pyplot as plt
import matplotlib.lines as pltlines
import pandas as pd

import datetime as dt
import certifi
from pymongo import MongoClient
from pymongo.errors import PyMongoError

import os
import sys
# This line adds the parent directory to the module search path so that the Stratcona module can be seen and imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import stratcona
from stratcona.modelling.relmodel import TestDef, ExpDims
from stratcona.engine.inference import int_out_v
from stratcona.engine.minimization import minimize_jax


def minimization():
    def test_parab(x, x_off):
        return (x + x_off) ** 2

    minimum = minimize_jax(test_parab, {'x_off': jnp.array([2, 3, -1])}, (-50.0, 50.0))
    print(minimum)


def jax_while():
    ins = (jnp.array(2), jnp.array(0))
    def stop(ins):
        return jnp.less(ins[1], 7)
    def accum(ins):
        return ins[0] + 2, ins[1] + 1
    res = jax.lax.while_loop(stop, accum, ins)
    print(res)


def dissertation_dist_transform_figure():
    mb = stratcona.SPMBuilder(mdl_name='VarTransform')

    var_tf = dists.transforms.ComposeTransform([dists.transforms.SoftplusTransform(), dist.transforms.AffineTransform(0, 0.1)])
    mb.add_params(obs_var=0.001)
    mb.add_hyperlatent('x1', dists.SoftLaplace, {'loc': 1.3, 'scale': 0.3})
    mb.add_observed('y1', dists.Normal, {'loc': 'x1', 'scale': 'obs_var'}, 1)
    mb.add_hyperlatent('x2', dists.SoftLaplace, {'loc': 1.3, 'scale': 0.3}, transform=dist.transforms.SoftplusTransform())
    mb.add_observed('y2', dists.Normal, {'loc': 'x2', 'scale': 'obs_var'}, 1)
    mb.add_hyperlatent('x3', dists.SoftLaplace, {'loc': 13, 'scale': 3.0}, transform=var_tf)
    mb.add_observed('y3', dists.Normal, {'loc': 'x3', 'scale': 'obs_var'}, 1)
    mb.add_hyperlatent('x4', dists.SoftLaplace, {'loc': 13, 'scale': 3.0}, transform=dist.transforms.SoftplusTransform())
    mb.add_observed('y4', dists.Normal, {'loc': 'x4', 'scale': 'obs_var'}, 1)

    print(dists.SoftLaplace(1.3, 0.3).variance)

    am = stratcona.AnalysisManager(mb.build_model(), rng_seed=26523646)
    am.set_test_definition(TestDef('blank', {'e': {'lot': 1, 'chp': 1}}, {'e': {}}))

    vals = am.sim_test_meas_new((1_000_000,))
    m1, m2, m3 = jnp.mean(vals['e_y1']), jnp.mean(vals['e_y2']), jnp.mean(vals['e_y3'])
    print(m1)
    print(m2)

    # Now plot
    seaborn.set_context('notebook', font_scale=1.25)
    seaborn.set_theme(style='ticks', font='Times New Roman')
    fig, p = plt.subplots()
    p.hist(vals['e_y1'].flatten(), bins=200, density=True, histtype='stepfilled', color='mediumblue', alpha=1.0, label='No transform')
    p.hist(vals['e_y2'].flatten(), bins=200, density=True, histtype='stepfilled', color='darkviolet', alpha=0.8, label='Softplus only')
    p.hist(vals['e_y3'].flatten(), bins=200, density=True, histtype='stepfilled', color='deeppink', alpha=0.7, label='Softplus + affine')
    p.axvline(float(m1), color='mediumblue', linestyle='dotted')
    p.axvline(float(m2), color='darkviolet', linestyle='dotted')
    p.axvline(float(m3), color='deeppink', linestyle='dotted')
    p.annotate('Target Distribution:', (-0.35, 1.2), fontweight='bold')
    p.annotate('Soft Laplace\nlocation=1.3, scale=0.3, domain=(0, ∞)', (-0.35, 1.0))

    p.set_xlim([-0.5, 3])
    p.set_xlabel('Variable Value')
    p.set_ylabel('Probability Density')
    p.set_yticks([])

    hndl, lbls = p.get_legend_handles_labels()
    lgnd1 = pltlines.Line2D([0], [0], color='black', linestyle='dotted')
    hndl.insert(3, lgnd1)
    lbls.insert(3, 'Distribution means')

    # Add the custom legend
    leg = p.legend(hndl, lbls, loc='upper right')
    for lbl in leg.legend_handles:
        lbl.set_alpha(1)

    plt.show()


### DATABASE CONNECTION HANDLING ###

DB_NAME = 'stratcona'
COLL_NAME = 'sandbox'

def login_to_database():
    tls_ca = certifi.where()
    uri = "mongodb+srv://arbutus.6v6mkhr.mongodb.net/?authSource=%24external&authMechanism=MONGODB-X509&retryWrites=true&w=majority"
    mongo_client = MongoClient(uri, tls=True, tlsCertificatekeyFile='cert/mongo_cert.pem', tlsCAFile=tls_ca)
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


def database_storage():
    dataset = login_to_database()
    test_data = {'f1': 4.1, 'f2': 'hello again', 'time': dt.datetime.now(tz=dt.UTC)}
    try_database_upload(dataset, test_data)


def examine_qual_data():
    ds = []
    for i in range(50):
        with open(f'data/bed_qual_y5k_x50k_batch{i}.json', 'r') as f:
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
                 #and e['v1'] == slice_vdd and e['v2'] == slice_vdd]
    t1 = t2 = [303.15, 328.15, 353.15, 378.15, 403.15]
    v1 = v2 = [0.8, 0.85, 0.9, 0.95, 1.0]
    u = jnp.array([d['u'] for d in slice]).reshape((5, 5))

    fig, ax = plt.subplots()
    ax.contourf(v1, v2, u)
    ax.set_ylabel('v1')
    ax.set_xlabel('v2')
    plt.show()


def examine_data():
    with open('bed_data/idfbcamp_bed_evals_y2k_x25k.json', 'r') as f:
        eigs = json.load(f)

    eig_list = [e for e in eigs.values()]
    eig_max = 0
    i_eig_max = -1
    for i, e in enumerate(eig_list):
        if e['EIG'] > eig_max:
            eig_max = e['EIG']
            i_eig_max = i
    print(eig_list[i_eig_max])
    high_eigs = [e for e in eig_list if e['EIG'] > eig_max * 0.99]

    slice_time, slice_vdd, slice_temp = 730, 1.0, 403.15
    eig_slice = [e for e in eigs.values() if e['T1'] == slice_time and e['T2'] == slice_time
                 and e['C1'] == slice_temp and e['C2'] == slice_temp]
                 #and e['V1'] == slice_vdd and e['V2'] == slice_vdd]
    t1 = t2 = [303.15, 328.15, 353.15, 378.15, 403.15]
    v1 = v2 = [0.8, 0.85, 0.9, 0.95, 1.0]
    eig = jnp.array([e['EIG'] for e in eig_slice]).reshape((5, 5))

    fig, ax = plt.subplots()
    ax.contourf(v1, v2, eig)
    ax.set_ylabel('V1')
    ax.set_xlabel('V2')
    plt.show()


def noise_testing():
    k = rand.key(28437)
    k, ky = rand.split(k, 2)
    s1, s2 = 3, 6
    y = dists.Normal(4, s1).sample(ky, (100_000,))
    print(f'Y mean - {jnp.mean(y)}, std dev - {jnp.std(y)}, min - {jnp.min(y)}, max - {jnp.max(y)}')
    yp = y.copy()
    k, kn = rand.split(k, 2)
    n = dists.Normal(0, s2).sample(kn, (100_000,))
    yp = yp + n
    print(f'YP mean - {jnp.mean(yp)}, std dev - {jnp.std(yp)}, min - {jnp.min(yp)}, max - {jnp.max(yp)}')
    print(f'Expected std dev: {jnp.sqrt((s1**2) + (s2**2))}')
    des = jnp.sqrt((jnp.std(yp)**2 - s2**2))
    y_rec = yp * (des / jnp.std(yp))
    print(f'Yrec mean - {jnp.mean(y_rec)}, std dev - {jnp.std(y_rec)}, min - {jnp.min(y_rec)}, max - {jnp.max(y_rec)}')
    print(f'Expected std dev: {s1}')


def simplest_lp_y_g_x():
    gold = 1.2
    y = {'e_y': jnp.array([[[gold]]])}
    var_tf = dists.transforms.ComposeTransform([dists.transforms.SoftplusTransform(), dists.transforms.AffineTransform(0, 0.1)])
    mb = stratcona.SPMBuilder('barebones')
    mb.add_hyperlatent('x', dists.Normal, {'loc': 1.3, 'scale': 0.0001})
    mb.add_hyperlatent('xs', dists.Normal, {'loc': 1, 'scale': 0.0001}, var_tf)
    mb.add_latent('v', nom='x', dev='xs')
    mb.add_params(ys=0.04)
    mb.add_observed('y', dists.Normal, {'loc': 'v', 'scale': 'ys'}, 100)

    am = stratcona.AnalysisManager(mb.build_model(), rng_seed=48)
    d = TestDef('bare', {'e': {'lot': 1, 'chp': 1}}, {'e': {}})
    am.set_test_definition(d)
    batch_dims = (1000, 500, 1)
    k = rand.key(3737)
    k1, k2, k3 = rand.split(k, 3)
    y_s = am.relmdl.sample_new(k1, d.dims, d.conds, (), am.relmdl.observes)
    print(f'Mean - {jnp.mean(y_s["e_y"])}, dev - {jnp.std(y_s["e_y"])}')

    am.relmdl.hyl_beliefs = {'x': {'loc': 1.2, 'scale': 0.2}, 'xs': {'loc': 0.9, 'scale': 0.2}}
    #x_s = am.relmdl.sample_new(k2, d.dims, d.conds, (batch_dims[0],), am.relmdl.hyls)
    ##x_s = {'x': jnp.full_like(x_s['x'], 1.3), 'xs': jnp.full_like(x_s['xs'], 0.1)}
    #lp, stats = int_out_v(k3, am.relmdl, batch_dims, d.dims, d.conds, x_s, y_s, {'y': 0.04})
    #print(stats)
    #p = jnp.exp(lp - jnp.max(lp))
    #print(f'P mean - {jnp.mean(p)}, std dev - {jnp.std(p)}, min - {jnp.min(p)}, max - {jnp.max(p)}')

    #df = pd.DataFrame({'lp': lp.flatten() - jnp.max(lp), 'x': x_s['x'], 'xs': x_s['xs']})
    #seaborn.scatterplot(df, x='x', y='xs', palette='viridis', hue='lp', hue_norm=(-10, 0))
    #plt.grid()
    #plt.show()

    y = {'e': {'y': y_s['e_y']}}
    perf = am.do_inference_is(y, n_x=1000)
    print(perf)
    print(am.relmdl.hyl_beliefs)

    # Now compare to HMC
    am.relmdl.hyl_beliefs = {'x': {'loc': 1.2, 'scale': 0.2}, 'xs': {'loc': 0.9, 'scale': 0.2}}
    am.do_inference(y)
    print(am.relmdl.hyl_beliefs)


def jit_behaviour():

    @partial(jax.jit, static_argnames=['op'])
    def sw_f(x, y, op):
        if op == 'add':
            return x + y
        elif op == 'sub':
            return x - y
        else:
            return x * y

    @partial(jax.jit, static_argnames=['f'])
    def reduce(vals, f):
        res = vals[0]
        for i in range(1, len(vals)):
            res = f(res, vals[i])
        return res

    myarr = jnp.array([2,3,4,5,6])

    print(f"Call: {reduce(myarr, partial(sw_f, op='add'))}")
    print(f"Call: {reduce(myarr, partial(sw_f, op='sub'))}")
    print(f"Call: {reduce(myarr, partial(sw_f, op='mul'))}")


def main():
    key = rand.key(363)
    k1, k2, k3 = rand.split(key, 3)

    ####################################################
    # Define an NBTI empirical model within the SPM framework for inference
    ####################################################
    # Model provided in JEDEC's JEP122H as generally used NBTI degradation model, equation 5.3.1
    def bti_vth_shift_empirical(a0, e_aa, temp, vdd, alpha, time, k, n):
        return 1000 * (a0 * 0.001) * jnp.exp((e_aa * 0.01) / (k * temp)) * (vdd ** alpha) * (time ** (n * 0.1))

    mb = stratcona.SPMBuilder(mdl_name='bti-empirical')
    mb.add_params(k=8.617e-5, zero=0.0, meas_var=1, n_nom=2, alpha=3.5, n=2)

    # Initial parameters are simulating some data to then learn
    mb.add_hyperlatent('a0_nom', dists.Normal, {'loc': 5, 'scale': 0.01})
    mb.add_hyperlatent('e_aa_nom', dists.Normal, {'loc': 6, 'scale': 0.01})

    var_tf = dists.transforms.ComposeTransform([dists.transforms.SoftplusTransform(), dists.transforms.AffineTransform(0, 0.1)])
    mb.add_hyperlatent('a0_dev', dists.Normal, {'loc': 6, 'scale': 0.01}, transform=var_tf)
    mb.add_hyperlatent('a0_chp', dists.Normal, {'loc': 2, 'scale': 0.01}, transform=var_tf)
    mb.add_hyperlatent('a0_lot', dists.Normal, {'loc': 3, 'scale': 0.01}, transform=var_tf)

    mb.add_latent('a0', nom='a0_nom', dev='a0_dev', chp='a0_chp', lot='a0_lot')
    mb.add_latent('e_aa', nom='e_aa_nom', dev=None, chp=None, lot=None)

    mb.add_intermediate('dvth', bti_vth_shift_empirical)

    mb.add_observed('dvth_meas', dists.Normal, {'loc': 'dvth', 'scale': 'meas_var'}, 3)

    am = stratcona.AnalysisManager(mb.build_model(), rng_seed=9861823450)

    pri = {'a0_nom': {'loc': 4.0, 'scale': 1.0},
           'a0_dev': {'loc': 7, 'scale': 2},
           'a0_chp': {'loc': 5, 'scale': 2},
           'a0_lot': {'loc': 5, 'scale': 2},
           'e_aa_nom': {'loc': 5, 'scale': 2}}
    d = TestDef('t1', {'e': {'lot': 4, 'chp': 2}}, {'e': {'temp': 55 + 273.15, 'vdd': 0.88, 'time': 1000}})

    am.set_test_definition(d)
    am.relmdl.hyl_beliefs = pri
    s_f = am.relmdl.sample_new
    y_s = s_f(k2, d.dims, d.conds, (), am.relmdl.observes)
    y = {'e': {'dvth_meas': y_s['e_dvth_meas']}}
    am.do_inference_mhgibbs(y, beta=0.25)
    print(am.relmdl.hyl_beliefs)

    #x_s = s_f(k1, d.dims, d.conds, (100,), am.relmdl.hyls)
    #y_s = s_f(k2, d.dims, d.conds, (1,), am.relmdl.observes)
    #get_lp_y_g_x = stratcona.engine.inference.int_out_v
    #start = time.time()
    #lp_y_g_x, perf_stats = get_lp_y_g_x(k3, am.relmdl, (100, 100, 1), d.dims, d.conds, x_s, y_s)
    #print(f'Unjitted: {time.time() - start}s')
    #to_time = partial(get_lp_y_g_x, k3, am.relmdl, (100, 100, 1), d.dims, d.conds, x_s, y_s)
    #best_perf = min(timeit.Timer(to_time).repeat(repeat=10, number=10))
    #print(f'Jitted: {best_perf}s')


def tig_calc():
    k = rand.key(2362736)
    k, k1 = rand.split(k)

    x1 = dists.Normal(0, 2.71)
    s1 = x1.sample(k1, (1_000_000,))
    m1 = jnp.where(s1 > 0, 1, 0)
    print(f'Percent over 0: {100 * (jnp.count_nonzero(m1) / 1_000_000)}%')
    lp1 = x1.log_prob
    h1 = stratcona.engine.bed.entropy(s1, lp1, limiting_density_range=(-838.8608, 838.8607))
    print(f'H32: {h1} nats')

    k, k1 = rand.split(k)
    x1 = dists.Normal(3, 2.3408)
    #x1 = dists.Normal(1, 0.7803)
    s1 = x1.sample(k1, (1_000_000,))
    m1 = jnp.where(s1 > 0, 1, 0)
    print(f'Percent over 0: {100 * (jnp.count_nonzero(m1) / 1_000_000)}%')
    lp1 = x1.log_prob
    h1 = stratcona.engine.bed.entropy(s1, lp1, limiting_density_range=(-838.8608, 838.8607))
    print(f'H32: {h1} nats')


if __name__ == '__main__':
    examine_data()
