from .v0 import SparseLinearV0
from .v1 import SparseLinearV1
from .v2 import SparseLinearV2

SSB_LAYERS = {
    "ssb-v0": SparseLinearV0,
    "ssb-v1": SparseLinearV1,
    "ssb-v2": SparseLinearV2,
}

def get_sparse_linear(name: str):
    try:
        return SSB_LAYERS[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown SSB implementation {name!r}. Available: {', '.join(SSB_LAYERS)}") from exc
