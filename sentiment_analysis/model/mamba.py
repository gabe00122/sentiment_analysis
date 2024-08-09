import jax
import numpy as np
from jax import numpy as jnp, random
from flax import nnx
from einops import rearrange
from sentiment_analysis.model.ssd import ssd
from sentiment_analysis.model.embeddings import Embedder
from sentiment_analysis.util import count_params


class Mamba2Layer(nnx.Module):
    def __init__(
        self,
        d_model,
        d_state=64,
        d_conv=4,
        expand=2,
        headdim=128,
        ngroups=1,
        *,
        A_init_range=(1, 16),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        chunk_size=256,
        rngs: nnx.Rngs
    ):
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.headdim = headdim
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.chunk_size = chunk_size

        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nnx.Linear(self.d_model, d_in_proj, use_bias=False, rngs=rngs)

        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        # print(conv_dim)
        # print(d_conv)
        # print(self.d_inner)
        padding = d_conv - 1
        self.conv1d = nnx.Conv(
            in_features=conv_dim,
            out_features=conv_dim,
            kernel_size=d_conv,
            feature_group_count=conv_dim,
            padding=((padding, padding),),
            rngs=rngs,
        )

        dt_key = rngs.params()

        # print(self.nheads)
        dt = jnp.exp(
            random.uniform(dt_key, (self.nheads,)) * (jnp.log(dt_max) - jnp.log(dt_min))
            + jnp.log(dt_min)
        )
        # return

        dt = jnp.clip(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + jnp.log(-jnp.expm1(-dt))
        self.dt_bias = nnx.Param(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        # self.dt_bias._no_weight_decay = True

        # A parameter
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A_key = rngs.params()

        A = random.uniform(
            A_key, (self.nheads,), minval=A_init_range[0], maxval=A_init_range[1]
        )
        A_log = jnp.log(A)
        self.A_log = nnx.Param(A_log)
        # self.register_buffer("A_log", torch.zeros(self.nheads, dtype=torch.float32, device=device), persistent=True)
        # self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nnx.Param(jnp.ones(self.nheads))
        # self.D._no_weight_decay = True

        # Extra normalization layer right before output projection
        self.norm = nnx.RMSNorm(self.d_inner, epsilon=1e-5, rngs=rngs)
        self.out_proj = nnx.Linear(
            self.d_inner, self.d_model, use_bias=False, rngs=rngs
        )

    def __call__(self, u):
        batch, seqlen, dim = u.shape

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        A = -jnp.exp(self.A_log.value)  # (nheads) or (d_inner, d_state)

        i = self.d_inner
        ii = self.d_inner + 2 * self.ngroups * self.d_state
        iii = self.nheads
        z, xBC, dt = jnp.split(zxbcdt, (i, i + ii), axis=-1)
        dt = jax.nn.softplus(dt + self.dt_bias.value)  # (B, L, nheads)

        # 1D Convolution
        # print(xBC.transpose().shape)
        xBC = jax.nn.silu(
            self.conv1d(xBC)
        )  # (B, L, self.d_inner + 2 * ngroups * d_state)
        xBC = xBC[:, :seqlen, :]

        # Split into 3 main branches: X, B, C
        # These correspond to V, K, Q respectively in the SSM/attention duality
        i = self.d_inner
        ii = self.ngroups * self.d_state
        iii = self.ngroups * self.d_state
        x, B, C = jnp.split(xBC, (i, i + ii), axis=-1)

        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim) * dt[..., jnp.newaxis]
        y, _ = ssd(
            x,
            A * dt,
            rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
            rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
            block_len=self.chunk_size,
        )
        y += x * self.D.value[np.newaxis, np.newaxis, :, np.newaxis]
        y = rearrange(y, "b l h p -> b l (h p)")

        # Multiply "gate" branch and apply extra normalization layer
        y = self.norm(y * z)
        out = self.out_proj(y)

        return out


class ResidualBlock(nnx.Module):
    def __init__(self, d_model, block, *, rngs: nnx.Rngs):
        self.norm = nnx.RMSNorm(d_model, rngs=rngs)
        self.block = block

    def __call__(self, x):
        return x + self.block(self.norm(x))


class Mamba2Model(nnx.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        d_model: int,
        d_state=64,
        d_conv=4,
        expand=2,
        headdim=128,
        ngroups=1,
        *,
        A_init_range=(1, 16),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        chunk_size=64,
        rngs: nnx.Rngs
    ):
        self.embedder = Embedder(
            vocab_size, d_model, dtype=jnp.float32, param_dtype=jnp.float32, rngs=rngs
        )

        self.layers = []
        for _ in range(num_layers):
            self.layers.append(
                ResidualBlock(
                    d_model=d_model,
                    block=Mamba2Layer(
                        d_model=d_model,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand,
                        headdim=headdim,
                        ngroups=ngroups,
                        A_init_range=A_init_range,
                        dt_min=dt_min,
                        dt_max=dt_max,
                        dt_init_floor=dt_init_floor,
                        chunk_size=chunk_size,
                        rngs=rngs,
                    ),
                    rngs=rngs,
                )
            )

        self.output_norm = nnx.RMSNorm(
            d_model, dtype=jnp.float32, param_dtype=jnp.float32, rngs=rngs
        )

    def __call__(self, x, _):
        x = self.embedder.encode(x)

        for layer in self.layers:
            x = layer(x)

        x = self.output_norm(x)
        x = self.embedder.decode(x)

        return x


def main():
    rngs = nnx.Rngs(0)
    mamba = Mamba2Model(32000, 12, 768, rngs=rngs)
    param_count = count_params(mamba)
    print(param_count)

    fake_key = rngs.fake()
    fake_input = random.randint(
        fake_key, (2, 128), 0, 32000, dtype=jnp.int16
    )  # jnp.ones((2, 128), dtype=jnp.int16)
    output = mamba(fake_input, None)

    print(output)


if __name__ == "__main__":
    main()
