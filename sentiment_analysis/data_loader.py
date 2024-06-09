import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from jax import numpy as jnp


def convert():
    table = pq.read_table("data/training_data.parquet")
    print(table)
    tokens = table["tokens_data"].combine_chunks()
    tokens_np = tokens.flatten().to_numpy().reshape((-1, 128))
    starts = table["stars"].to_numpy()
    lengths = table["length"].to_numpy()

    np.savez_compressed(
        "data/training_data.npz", tokens=tokens_np, starts=starts, lengths=lengths
    )
    # tokens_np = np.zeros((), dtype=np.int16)


def main():
    data = jnp.load("data/training_data.npz")
    tokens = data["tokens"]
    print(data)
    print(tokens.dtype)


if __name__ == "__main__":
    main()
