from pathlib import Path
import random

import torch
import audiotools as at
from tqdm import tqdm


INPUT_DIR = Path("/home/mila/n/nithya.shikarpur/scratch/cat-rave/data/")
OUTPUT_DIR = Path("/home/mila/n/nithya.shikarpur/scratch/cat-rave/data-cat-bass/")

MIN_DURATION = 10.0 # min duration of each file in seconds

def concat_if_under_min_duration(input_dir: Path, output_dir: Path, min_duration: float):
    """
    Concatenate audio files in the input directory if they are under the minimum duration.
    NOTE: will downsample all signals to mono
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    audio_files = at.util.find_audio(input_dir)
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
    concat_if_under_min_duration(INPUT_DIR, OUTPUT_DIR, MIN_DURATION)