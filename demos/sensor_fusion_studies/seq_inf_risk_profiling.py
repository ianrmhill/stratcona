# Copyright (c) 2025 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import numpyro
import numpyro.distributions as dists
# This call has to occur before importing jax
numpyro.set_host_device_count(4)

import jax.numpy as jnp
import jax.random as rand

import os
import sys
# This line adds the parent directory to the module search path so that the Stratcona module can be seen and imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import stratcona


def live_risk_profiling():
    pass

    ### Available sensors in peripheral: ###
    # Generic sensors: Temp, Vdd, Standard RO
    # Specialized RO sensors: PBTI, NBTI, n-HCI, p-HCI
    # Specialized induced offset sensors: PBTI, NBTI, n-HCI, p-HCI
    # Specialized staggered EM sensor
    # Two TDDB sensors: Frequency based, Ramp voltage

    # Note that in general this study should be reasonably short, as the paper can't get too long and it is just a demo

    # NOTE: Wear-out Sensor Fusion SoC Peripheral for Continuous Semiconductor Reliability Monitoring, Lifespan Prediction, and Failure Risk Management
    # Based on title, this study should illustrate continuous monitoring, remaining useful life prediction, and imminent failure risk to be fully successful

    # Use repeated temp and voltage measurements to build an effective total stress estimate over time

    # Start with a segmented model inferred from qualification tests, including chip-to-chip and device-to-device variability

    # For each individual in-field unit/chip, ongoing telemetry is used to try and determine where in the chip-to-chip
    # variability spread the part lies

    # One model output is a next-day probability of failure, based on the QX%-LBCI. The portion of the distribution to
    # the left of one day from now (including in the past) is summed to get the probability of failure. PROBLEM: Model
    # should not include any past probability density, handle this correctly from a statistical perspective.

    # Another model output is expected remaining useful lifespan, used for prediction

    # For this study, keep dynamic stress really simple. Assume constant stress throughout life, but the probability of
    # failure can potentially consider under different conditions. Based on the current amount of degradation, inverse
    # compute the time required to reach that amount of degradation for each considered upcoming stress condition, then
    # compute the probability of within next day failure.

    # Discuss using the current probability of failure to tune voltage and frequency margins


if __name__ == '__main__':
    live_risk_profiling()
