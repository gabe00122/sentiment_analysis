from jax import numpy as jnp, random
from einops import rearrange, repeat


def segsum_unstable(x):
    """Naive segment sum calculation. exp(segsum(A)) produces a 1-SS matrix,
       which is equivalent to a scalar SSM."""
    T = x.shape[-1]
    x_cumsum = jnp.cumsum(x, axis=-1)
    x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
    mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool), k=0)
    x_segsum = jnp.where(mask, x_segsum, -jnp.inf)
    return x_segsum


def segsum(x):
    """More stable segment sum calculation."""
    T = x.shape[-1]
    x = repeat(x, "... d -> ... d e", e=T)
    mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool), -1)
    x = jnp.where(mask, x, 0)
    x_segsum = jnp.cumsum(x, -2)
    mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool))
    x_segsum = jnp.where(mask, x_segsum, -jnp.inf)
    return x_segsum


def ssd(X, A, B, C, block_len=64, initial_states=None):
    """
    Arguments:
        X: (batch, length, n_heads, d_head)
        A: (batch, length, n_heads)
        B: (batch, length, n_heads, d_state)
        C: (batch, length, n_heads, d_state)
    Return:
        Y: (batch, length, n_heads, d_head)
    """
    assert X.dtype == A.dtype == B.dtype == C.dtype
    assert X.shape[1] % block_len == 0

    # Rearrange into blocks/chunks
    X, A, B, C = [rearrange(x, "b (c l) ... -> b c l ...", l=block_len) for x in (X, A, B, C)]

    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = jnp.cumsum(A, axis=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = jnp.exp(segsum(A))
    Y_diag = jnp.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    decay_states = jnp.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = jnp.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    if initial_states is None:
        initial_states = jnp.zeros_like(states[:, :1])

    states = jnp.concatenate([initial_states, states], axis=1)
    decay_chunk = jnp.exp(segsum(jnp.pad(A_cumsum[:, :, :, -1], ((0, 0), (0, 0), (1, 0)))))

    new_states = jnp.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    state_decay_out = jnp.exp(A_cumsum)
    Y_off = jnp.einsum('bclhn,bchpn,bhcl->bclhp', C, states, state_decay_out)

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    Y = rearrange(Y_diag+Y_off, "b c l h p -> b (c l) h p")
    return Y, final_state




def main():
    batch, seqlen, chunk_size, dim, headdim = 1, 2048, 64, 2048, 64
    nheads = dim // headdim  # (H) in the paper
    ngroups = 1 # (G) in the paper
    dstate = 64  # (N) in the paper

    seed = random.PRNGKey(0)
    x_key, a_key, b_key, c_key = random.split(seed, 4)

    X = random.normal(x_key, (batch, seqlen, nheads, headdim))
    A = -jnp.exp(random.normal(a_key, (batch, seqlen, nheads)))
    B = random.normal(b_key, (batch, seqlen, nheads, dstate))
    C = random.normal(c_key, (batch, seqlen, nheads, dstate))

    Y, _ = ssd(X, A, B, C)
    print(Y)


if __name__ == '__main__':
    main()
