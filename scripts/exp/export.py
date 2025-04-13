from pathlib import Path
import shutil

import argparse

parser = argparse.ArgumentParser(description="Export the fine-tuned model to the repo")
parser.add_argument(
    "--name", type=str, default="lazaro-ros-sep",
    help="name of the fine-tuned model to export"
)
parser.add_argument(
    "--model", type=str, default="latest",
    help="model version to export. check runs/<name> for available versions"
)

parser.add_argument(
    "--run_dir", type=str, default="runs",
    help="directory where the run is stored"
)

args = parser.parse_args()
name = args.name
version = args.model

run_dir = Path(f"{args.run_dir}/{name}")
repo_dir = Path("models/vampnet")

for part in ("coarse", "c2f"):
    outdir = repo_dir / "loras" / name 
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"{part}.pth"
    path = run_dir / part / version / "vampnet" / "weights.pth"
    shutil.copy(path, outpath)
    print(f"moved {path} to {outpath}")

from huggingface_hub import Repository
repo = Repository(str(repo_dir))
print(f"pushing {repo_dir} to {name}")
repo.push_to_hub(
    commit_message=f"add {name}",
)
print("done!!! >::0")