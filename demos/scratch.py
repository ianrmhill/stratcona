import numpy as np
import pytensor
import pytensor.tensor as pt
import pymc

import os
import sys
# This line adds the parent directory to the module search path so that the Stratcona module can be seen and imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stratcona.assistants.probability import shorthand_compile
from stratcona.assistants.iterator import iter_sampler
from stratcona.engine.boed import entropy, information_gain
from stratcona.engine.inference import inference_model, fit_latent_params_to_posterior_samples

if __name__ == '__main__':
    with pymc.Model() as mdl:
        a = pymc.Normal('a', mu=0, sigma=0.1, shape=(5,))
        out = pymc.Normal('out', mu=2 * a, sigma=0.02, observed=[0.2, 0.1, 0.15, 0.1, 0.2])

    gv = pymc.model_to_graphviz(mdl)
    gv.format = 'png'
    gv.render(filename='model_graph')

    with mdl:
        pymc.sample()


def boed_testing():
    with pymc.Model() as mdl:
        a = pymc.Normal('a', mu=0, sigma=0.1)
        out = pymc.Normal('out', mu=2 * a, sigma=0.02)

    prior_sampler = shorthand_compile('ltnt_sampler', mdl, [a], [out], [])
    prior_val_iter = iter_sampler([[0], [1]])
    p_prior = shorthand_compile('ltnt_logp', mdl, [a], [out], [])
    p_obs = shorthand_compile('obs_logp', mdl, [a], [out], [])

    h_a = entropy(prior_sampler, p_prior, in_bits=True)
    h_a_ld = entropy(prior_sampler, p_prior, in_bits=True, limiting_density=True)
    print(f"Entropy in 'a': {h_a}, limiting density version: {h_a_ld}")

    obs = pytensor.shared(0.152)
    with pymc.Model() as mdl2:
        a2 = pymc.Normal('a2', mu=0, sigma=0.1)
        out2 = pymc.Normal('out2', mu=2 * a2, sigma=0.02, observed=obs)
    inf_samples = inference_model(mdl2)
    ltnt_post = fit_latent_params_to_posterior_samples([a2], {'a2': ['mu', 'sigma']}, inf_samples)
    print(ltnt_post)

    with pymc.Model() as mdl3:
        a3 = pymc.Normal('a3', mu=ltnt_post['a2']['mu'], sigma=ltnt_post['a2']['sigma'])
        out3 = pymc.Normal('out3', mu=2 * a3, sigma=0.02)
    post_sampler = shorthand_compile('ltnt_sampler', mdl3, [a3], [out3], [])
    p_post = shorthand_compile('ltnt_logp', mdl3, [a3], [out3], [])

    h_a = entropy(post_sampler, p_post, in_bits=True)
    h_a_ld = entropy(post_sampler, p_post, in_bits=True, limiting_density=True)
    print(f"Entropy in 'a': {h_a}, limiting density version: {h_a_ld}")

    ig = information_gain(prior_sampler, p_prior, p_post, in_bits=True)
    ig_ld = information_gain(prior_sampler, p_prior, p_post, in_bits=True, limiting_density=True)
    print(f"Information gain of observing {obs.get_value()}: {ig}, limiting density version: {ig_ld}")

    obs.set_value(0.97)
    inf_samples = inference_model(mdl2)
    ltnt_post = fit_latent_params_to_posterior_samples([a2], {'a2': ['mu', 'sigma']}, inf_samples)
    print(ltnt_post)

    with pymc.Model() as mdl4:
        a4 = pymc.Normal('a4', mu=ltnt_post['a2']['mu'], sigma=ltnt_post['a2']['sigma'])
        out4 = pymc.Normal('out4', mu=2 * a4, sigma=0.02)
    post_sampler = shorthand_compile('ltnt_sampler', mdl4, [a4], [out4], [])
    p_post = shorthand_compile('ltnt_logp', mdl4, [a4], [out4], [])

    h_a = entropy(post_sampler, p_post, in_bits=True)
    h_a_ld = entropy(post_sampler, p_post, in_bits=True, limiting_density=True)
    print(f"Entropy in 'a': {h_a}, limiting density version: {h_a_ld}")

    # TODO: BOED sampling stats to show that this doesn't work well
    ig = information_gain(prior_sampler, p_prior, p_post, in_bits=True)
    ig_ld = information_gain(prior_sampler, p_prior, p_post, in_bits=True, limiting_density=True)
    print(f"Information gain of observing {obs.get_value()}: {ig}, limiting density version: {ig_ld}")
