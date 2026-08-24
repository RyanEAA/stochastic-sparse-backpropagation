from .v0 import SparseLinearV0
from .v1 import SparseLinearV1
from .v2 import SparseLinearV2
from .v3 import SparseLinearV3
from .v1_block import BlockSparseLinearV1
from .v2_block import BlockSparseLinearV2
from .v3_block import BlockSparseLinearV3

SSB_LAYERS = {
    "ssb-v0": SparseLinearV0,
    "ssb-v1": SparseLinearV1,
    "ssb-v2": SparseLinearV2,
    "ssb-v3": SparseLinearV3,
    "ssb-v1-block": BlockSparseLinearV1,
    "ssb-v2-block": BlockSparseLinearV2,
    "ssb-v3-block": BlockSparseLinearV3,
}


def get_sparse_linear(name: str):
    try:
        return SSB_LAYERS[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown SSB implementation {name!r}. Available: {', '.join(SSB_LAYERS)}") from exc
