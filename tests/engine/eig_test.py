# Copyright (c) 2024 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from stratcona.engine.boed import eig_smc_joint_likelihood


def test_eig_smc():
    # Given these four definitions, lp_i will always return -2.2, lp_y always returns -4.5
    def i_s():
        return [np.array([0.5]), np.array([1])]
    def y_s():
        return [np.array([[2]]), np.array([[3]])]
    def lp_i(i0, i1):
        return -1.2*i1 + -2*i0
    def lp_y(i0, i1, y0, y1):
        return -3*i0*i1*y0 + -1*i0*y1

    rng = np.random.default_rng(44)
    eig, mig = eig_smc_joint_likelihood(5, 5, i_s, y_s, lp_i, lp_y, 3, False, False, rng)
    assert round(eig, 5) == 0
    assert round(mig, 5) == 0
