import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Any, NamedTuple

from human_id import generate_id  # pip install human-id

from nvml.xr_helpers import WindDirectionBinNames
from nvml.rprep.collate import (
    RegDataPaths,
    collate_data,
    prepare_core_feature_df,
    prepare_pca_df,
    prepare_qoi_df,
)
from nvml.flearn.pca.make_pca import PCAId


class DataSpec(NamedTuple):  # Or CaseInputs / PrepContext
    flow_qoi: Path
    pca_path: Path
    rdp: RegDataPaths
    n: int
    drn: WindDirectionBinNames
    pcaid: PCAId

    def to_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, tuple) and hasattr(value, "_asdict"):
                return {k: convert(v) for k, v in value._asdict().items()}
            return value

        return {k: convert(v) for k, v in self._asdict().items()}

    def run_slug(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True).encode()
        seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        readable_id = generate_id(seed=seed, word_count=3)  # e.g., "appear-hard-idea"

        drn_val = self.drn.value if isinstance(self.drn, Enum) else self.drn
        return f"n{self.n}_{drn_val}_{self.pcaid.path_name}_{readable_id}"


def prepare_data(
    flow_qoi: Path,
    pca_path: Path,
    rdp: RegDataPaths,
    n: int,
    drn: WindDirectionBinNames,
    pcaid: PCAId,
):

    qoi_df = prepare_qoi_df(flow_qoi)
    pca_df = prepare_pca_df(pca_path, pcaid, n)
    rdp = RegDataPaths(*rdp)
    cdf = prepare_core_feature_df(rdp, drn)

    df = collate_data(cdf, pca_df, qoi_df, drn)
    return df
