# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

from pytensor.tensor import TensorVariable
import pymc

__all__ = ['shorthand_compile']


def shorthand_compile(which_func: str, model: pymc.Model,
                      ltnt_vars: list[TensorVariable], obs_vars: list[TensorVariable], pred_vars: list[TensorVariable]):
    # The most complicated part of understanding model compilation is differentiating the three different graph nodes
    # associated with each random variable. Each random variable can be randomly sampled, can have a specific value, and
    # has a specific log probability of taking any specific value. The correct nodes must be provided to the PyMC
    # compiler, not just the correct random variable. The random sample node is accessed by specifying the random
    # variable itself, the specific value nodes are found via 'model.rvs_to_values[<random variable>]', and the log
    # probability via 'model.logp(vars=<random variable>)'. There are other model attributes that provide alternate ways
    # of obtaining these nodes, but the previously listed ones are consistent. It can be quite confusing and hopefully
    # PyMC will make these distinctions clearer in future versions.
    ltnt_setpoint_nodes = [model.rvs_to_values[var] for var in ltnt_vars]
    obs_setpoint_nodes = [model.rvs_to_values[var] for var in obs_vars]
    ins, outs = [], []
    if which_func == 'ltnt_sampler':
        # No need for inputs, just get latent variable sample outputs based on the prior distribution
        outs = ltnt_vars
    elif which_func == 'obs_sampler':
        # No need for inputs, just get latent variable sample outputs based on the prior distribution
        outs = obs_vars
    elif which_func == 'ltnt_logp':
        # Need to input the specific values that were sampled for the latent variables, output the logp of those values
        ins = ltnt_setpoint_nodes
        outs = model.logp(vars=ltnt_vars)
    elif which_func == 'obs_logp':
        # Need to also input the specific values for the observed variables, output the logp of the observations
        ins = ltnt_setpoint_nodes + obs_setpoint_nodes
        outs = model.logp(vars=obs_vars)
    elif which_func == 'life_sampler':
        outs = pred_vars
    else:
        raise NotImplementedError

    # With the correct input and output lists we can tell PyMC to compile the requested function
    return pymc.compile_pymc(ins, outs, name=which_func)
