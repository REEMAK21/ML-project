import typer
from pathlib import Path
from .train import run_train
from .config import TrainCfg

app = typer.Typer()

@app.command()
def train(
    features: Path = typer.Argument(..., help="Path to features CSV"),
    target: str = typer.Argument(..., help="Target column name")
):
    cfg = TrainCfg(
        features_path=features,
        target=target
    )
    run_train(cfg)


if __name__ == "__main__":
    root = Path(".")
    cfg = TrainCfg(
        features_path=Path("data/features/train.parquet"),
        target="is_high_value"
    )
    run_train(cfg, root=root)
    print("Training complete. Check models/runs/")
    
