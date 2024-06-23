
def get_fixup_scale(transformer_layers: int, fixup_constant: float = 9) -> float:
    return (fixup_constant * transformer_layers) ** -(1 / 4)


def get_embed_scale(embedding_features: int) -> float:
    return embedding_features ** -(1 / 2)
