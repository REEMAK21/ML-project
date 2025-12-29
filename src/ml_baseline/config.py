from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    root: Path
    raw: Path
    cache: Path
    processed: Path
    external: Path
    reports: Path

    @classmethod
    def from_repo_root(cls) -> "Paths":
        """
        Automatically find the repo root (where pyproject.toml is)
        and return a Paths object with proper folder paths.
        """
        root = Path(__file__).parent.parent.parent.resolve()
        data = root / "data"
        reports = root / "reports"  # reports folder outside data
        return cls(
            root=root,
            raw=data / "raw",
            cache=data / "cache",
            processed=data / "processed",
            external=data / "external",
            reports=reports
        )

def make_paths(root: Path) -> Paths:
    data = root / "data"
    reports = root / "reports"
    return Paths(
        root=root,
        raw=data / "raw",
        cache=data / "cache",
        processed=data / "processed",
        external=data / "external",
        reports=reports
    )


@dataclass(frozen=True)
class TrainCfg:
    features_path: Path
    target: str

    # columns
    id_cols: tuple[str, ...] = ("id",)          # optional passthrough identifiers
    time_col: str | None = None                # if set, sort by time before splitting
    group_col: str | None = None               # if set, use group-safe splitting

    # binary classification specifics
    pos_label: str | int = 1                   # define your positive class
    threshold_strategy: str = "fixed"          # "fixed" | "min_precision" | "max_f1"
    threshold_value: float = 0.5
    min_precision: float = 0.8

    # splitting / reproducibility
    session_id: int = 42
    train_size: float = 0.8
    fold: int = 5

    # model selection
    sort_metric: str = "AUC"     # PyCaret metric name
    tune_metric: str = "AUC"
    tune_iters: int = 25


