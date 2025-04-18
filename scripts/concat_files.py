from pathlib import Path
import random

import torch
import audiotools as at
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="Concatenate audio files to a minimum duration")
parser.add_argument(
    "--input_dir", type=str, default="/home/mila/n/nithya.shikarpur/scratch/cat-rave/data/",
    help="parent input directory"
)
parser.add_argument(
    "--output_dir", type=str,
    help="output directory"
)
parser.add_argument(
    "--dataset_names", type=str, nargs="+",
    help="dataset names in parent directory to concatenate"
) 
args = parser.parse_args()
# INPUT_DIR = Path("/home/mila/n/nithya.shikarpur/scratch/cat-rave/data/")
# OUTPUT_DIR = Path("/home/mila/n/nithya.shikarpur/scratch/cat-rave/data-cat-bass/")

MIN_DURATION = 10.0 # min duration of each file in seconds

def concat_if_under_min_duration(input_dir: Path, output_dir: Path, min_duration: float, dataset_names: list):
    """
    Concatenate audio files in the input directory if they are under the minimum duration.
    NOTE: will downsample all signals to mono
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    audio_files = []
    for dataset_name in dataset_names:
        dataset_dir = input_dir / dataset_name
        audio_files += at.util.find_audio(dataset_dir)
        print(f"found {len(audio_files)} audio files in {dataset_dir}")
    sigs = [at.AudioSignal(af) for af in audio_files]

    for sig in tqdm(sigs):
        sig.to_mono()
        while sig.duration < min_duration:
            # get a random file from the list
            sig_to_concat = random.choice(sigs)
            sig_to_concat.to_mono()
            print(f"concatting {sig_to_concat.path_to_file} to {sig.path_to_file}")
            # norm the new sig to the orig sig
            sig.normalize(sig_to_concat.loudness())
            # concat the two signals
            sig.samples = torch.cat([sig.samples, sig_to_concat.samples], dim=-1)

    # save our new files
    for sig in sigs:
        outpath = output_dir / Path(sig.path_to_file).name
        sig.write(outpath)
        print(f"saved sig to {outpath}")

    # print total duration of all files
    total_duration = sum([sig.duration for sig in sigs])
    print(f"total duration of all files: {total_duration:.2f} seconds")

if __name__ == "__main__":
    INPUT_DIR = Path(args.input_dir)
    OUTPUT_DIR = Path(args.output_dir)
    dataset_names = args.dataset_names
    concat_if_under_min_duration(INPUT_DIR, OUTPUT_DIR, MIN_DURATION, dataset_names)