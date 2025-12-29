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