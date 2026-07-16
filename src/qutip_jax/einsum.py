import jax.numpy as jnp
from .jaxarray import JaxArray
import numpy as np
from qutip.core.data import einsum
from qutip.core.data.convert import to as _to

__all__ = []

def einsum_jax(
        op0, /,
        subscripts,
        rest_operands,
        tensor_shapes,
        tensor_perms,
        out_perm,
        out_shape=None
):
    """
    JAX / XLA specialization for einsum.
    """
    operands = (op0,) + tuple(rest_operands)
    tensors = []
    
    for op, shape, perm in zip(operands, tensor_shapes, tensor_perms):
        jax_op = _to(JaxArray, op)
        
        arr = jax_op._jxa
        tensors.append(jnp.transpose(jnp.reshape(arr, shape), perm))

    result = jnp.einsum(subscripts, *tensors)

    # Enforce 1x1 shape for Cython dispatcher compatibility on scalars
    if result.shape == ():
        return JaxArray(jnp.array([[result]], dtype=jnp.complex128))

    inv_out_perm = np.argsort(out_perm)
    result_physical = jnp.transpose(result, inv_out_perm)

    if out_shape is None:
        half = result_physical.ndim // 2
        rows = int(np.prod(result_physical.shape[:half]))
        cols = int(np.prod(result_physical.shape[half:]))
        out_shape = (rows, cols)

    return JaxArray(jnp.reshape(result_physical, out_shape))

einsum.add_specialisations([
    (JaxArray, JaxArray, einsum_jax),
])