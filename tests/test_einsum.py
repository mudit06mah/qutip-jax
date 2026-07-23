import numpy as np
import pytest
import qutip as qt
from qutip.core.dimensions import einsum
from qutip_jax import JaxArray, einsum_jax
from qutip.tests.core.data.test_einsum import TestEinsum as _TestEinsum

_cases_mark = [
    m for m in _TestEinsum.test_einsum.pytestmark
    if m.name == 'parametrize' and 'subscripts' in m.args[0]
][0]


class TestEinsum:
    specialisations = [
        pytest.param(einsum_jax, JaxArray, JaxArray, id="JaxArray"),
    ]

    @pytest.mark.parametrize("einsum_func, data_type, out_type", specialisations)
    @pytest.mark.parametrize(*_cases_mark.args)
    def test_einsum(
        self,
        einsum_func,
        data_type,
        out_type,
        subscripts,
        shapes,
        perms,
        out_perm,
        out_shape,
        operands_data,
        expected_data,
    ):
        _TestEinsum().test_einsum(
            einsum_func,
            data_type,
            out_type,
            subscripts,
            shapes,
            perms,
            out_perm,
            out_shape,
            operands_data,
            expected_data,
        )


_cx = qt.Qobj(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
    dims=[[2, 2], [2, 2]],
).to("jax")
_cx_dag = _cx.dag()
_rho_01 = qt.ket2dm(
    qt.tensor(qt.basis(2, 0), qt.basis(2, 1))
).to("jax")
_thermal_dm_2q = qt.tensor(
    qt.thermal_dm(2, 1), qt.thermal_dm(2, 1)
).to("jax")


@pytest.mark.parametrize(["subscripts", "operands", "expected"], [
    pytest.param("ii", [qt.sigmaz().to("jax")], 0),
    pytest.param("ij,ji", [qt.sigmaz().to("jax"), qt.sigmaz().to("jax")], 2),
    pytest.param(
        "ijij", [_thermal_dm_2q], 1,
    ),
    pytest.param(
        "ikjl,jm->ikml",
        [qt.tensor(qt.sigmaz(), qt.sigmaz()).to("jax"), qt.sigmaz().to("jax")],
        qt.tensor(qt.qeye(2), qt.sigmaz()).to("jax"),
    ),
    pytest.param(
        "abcd,cde->abe",
        [_cx, qt.tensor(qt.basis(2, 0), qt.basis(2, 1)).to("jax")],
        qt.tensor(qt.basis(2, 0), qt.basis(2, 1)).to("jax"),
        id="ket_multiplication",
    ),
    pytest.param(
        "abcd,cdef->abef",
        [_cx, _rho_01],
        _cx @ _rho_01,
        id="density_matrix_left_multiplication",
    ),
    pytest.param(
        "cdef,efgh->cdgh",
        [_rho_01, _cx_dag],
        _rho_01 @ _cx_dag,
        id="density_matrix_right_multiplication",
    ),
    pytest.param(
        "abcd,cdef,efgh->abgh",
        [_cx, _rho_01, _cx_dag],
        _cx @ _rho_01 @ _cx_dag,
        id="density_matrix_conjugation",
    ),
])
def test_qobj_einsum(subscripts, operands, expected):
    res = einsum(subscripts, *operands)
    if isinstance(expected, qt.Qobj):
        assert isinstance(res.data, JaxArray)
        assert res.dims == expected.dims
        np.testing.assert_allclose(res.full(), expected.full(), atol=1e-12)
    else:
        assert np.isclose(res, expected, atol=1e-12)


@pytest.mark.parametrize(["subscripts", "operands"], [
    pytest.param(
        "ij", [qt.sigmax().to("jax")],
        id="single_operand_no_contraction",
    ),
    pytest.param(
        "ij->ji", [qt.sigmay().to("jax")],
        id="single_operand_transpose",
    ),
    pytest.param(
        "ijkl->kjil",
        [qt.tensor(qt.sigmam(), qt.sigmaz()).to("jax")],
        id="single_operand_permutation",
    ),
    pytest.param(
        "cdef,ghef->cdgh",
        [_rho_01, _cx],
        id="col_col_contraction",
    ),
])
def test_einsum_rejects_implicit_transpose(subscripts, operands):
    with pytest.raises(ValueError):
        einsum(subscripts, *operands)


