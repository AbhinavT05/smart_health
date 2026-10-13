"""Single reproducible CLI for the MobiFall v2.0 research pipeline."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RAW = ROOT / "data" / "raw" / "MobiFall"


def _config(command: str) -> Path:
    out = ROOT / "reports" / "pipeline_configs"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{command}.json"
    path.write_text(json.dumps({"dataset": "MobiFall v2.0", "command": command, "seed": 42,
                                "outer_folds": 5, "generated_utc": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")
    print(f"Run config: {path}")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["inspect", "experiment", "train", "all"])
    args = parser.parse_args(argv)
    if not RAW.is_dir():
        raise FileNotFoundError(f"Real MobiFall v2.0 data is required at {RAW}")
    _config(args.command)
    if args.command in {"inspect", "all"}:
        from scripts.inspect_mobifall import main as inspect_main
        inspect_main()
    if args.command in {"experiment", "all"}:
        from scripts.run_mobifall_experiments import main as experiment_main
        experiment_main()
    elif args.command == "train":
        # Training starts from source trials and fits the serving bundle without a CSV input.
        from scripts.build_mobifall_manifest import main as manifest_main
        manifest_main()
        from scripts.run_mobifall_experiments import main as experiment_main
        experiment_main()
    if args.command in {"train", "all"}:
        from scripts.save_mobifall_bundle import main as save_main
        save_main()


if __name__ == "__main__":
    main()
