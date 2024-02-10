import numpy as np
import pytensor as pt
import pymc


prms = {'vdd': [0.2, 0.3], 'temp': [300, 350]}
for prm in prms:
    prms[prm] = np.expand_dims(prms[prm], axis=1)
print('Wow')


latents = pt.shared(np.ones((2, 2)))
obs = pt.shared(np.array([1.1, 2.0]))

with pymc.Model() as mdl:
    prms = latents
    a_prms = {'mu': prms[0][0], 'sigma': prms[0][1]}
    b_prms = {'mu': prms[1][0], 'sigma': prms[1][1]}
    a = pymc.Normal('a', **a_prms)
    b = pymc.Normal('b', **b_prms)

    c_mu = 0.1
    c = pymc.Normal('c', a + b, c_mu)
    c_obs = pymc.Normal('c_obs', a + b, c_mu, observed=obs[0])

ltnt_sampler = pymc.compile_pymc([], [a, b], name='ltnt_sampler')
obs_logp = pymc.compile_pymc([mdl.rvs_to_values[a], mdl.rvs_to_values[b], mdl.rvs_to_values[c]], mdl.logp(vars=c), name='obs_logp')
obs_logp_const = pymc.compile_pymc([mdl.rvs_to_values[a], mdl.rvs_to_values[b]], mdl.logp(vars=c_obs), name='obs_logp')

samples = np.array([ltnt_sampler() for _ in range(100)])
print(samples.mean())
print(obs_logp(0.4, 0.7, 1.1))
print(obs_logp_const(0.4, 0.7))
obs.set_value(1.5)
print(obs_logp(0.4, 0.7, 1.5))
print(obs_logp_const(0.4, 0.7))

latents.set_value([[0, 1], [4, 0.2]])
samples = np.array([ltnt_sampler() for _ in range(100)])
print(samples.mean())
