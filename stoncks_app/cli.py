from __future__ import annotations

import argparse
from pathlib import Path

from stoncks_app.app import launch_app
from stoncks_app.data import make_dataset
from stoncks_app.model import load_model, save_model, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NQ! alert application")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train a model")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--model", type=Path, required=True)

    run = sub.add_parser("run", help="Run the GUI")
    run.add_argument("--data", type=Path, required=True)
    run.add_argument("--model", type=Path, required=True)

    report = sub.add_parser("report", help="Print model report")
    report.add_argument("--model", type=Path, required=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        dataset = make_dataset(args.data)
        result = train_model(dataset)
        save_model(result.model, args.model)
        print(result.report)
    elif args.command == "run":
        launch_app(args.model, args.data)
    elif args.command == "report":
        model = load_model(args.model)
        print(model)


if __name__ == "__main__":
    main()
