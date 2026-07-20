import numpy as np
import pytest
import qutip as qt
from qutip.core.dimensions import einsum
from qutip_jax import JaxArray, einsum_jax
from qutip.tests.core.data.test_einsum import TestEinsum as _TestEinsum

class TestEinsum(_TestEinsum):
    specialisations = [
        pytest.param(einsum_jax, JaxArray, JaxArray, id="JaxArray"),
    ]

def test_qobj_einsum():
    # Setup JaxArray Qobjs
    psi1 = qt.basis(2, 0).to("jax")
    psi2 = qt.basis(2, 1).to("jax")
    psi = qt.tensor(psi1, psi2)
    
    # Define a CX gate manually
    cx_matrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=complex)
    cx = qt.Qobj(cx_matrix, dims=[[2, 2], [2, 2]]).to("jax")
    
    res = einsum('abcd,cde->abe', cx, psi)
    assert isinstance(res.data, JaxArray)
    
    expected = qt.tensor(qt.basis(2, 0), qt.basis(2, 1)).to("jax")
    np.testing.assert_allclose(res.full(), expected.full(), atol=1e-12)
