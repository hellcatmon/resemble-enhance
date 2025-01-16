import argparse
import random
import time
import torch.multiprocessing as mp
from pathlib import Path
import torch
from .parallel import process_files


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("in_dir", type=Path, help="Path to input audio folder")
    parser.add_argument("out_dir", type=Path, help="Output folder")
    parser.add_argument(
        "--run_dir",
        type=Path,
        default=None,
        help="Path to the enhancer run folder, if None, use the default model",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=".wav",
        help="Audio file suffix",
    )
    parser.add_argument(
        "--denoise_only",
        action="store_true",
        help="Only apply denoising without enhancement",
    )
    parser.add_argument(
        "--lambd",
        type=float,
        default=1.0,
        help="Denoise strength for enhancement (0.0 to 1.0)",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.5,
        help="CFM prior temperature (0.0 to 1.0)",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="midpoint",
        choices=["midpoint", "rk4", "euler"],
        help="Numerical solver to use",
    )
    parser.add_argument(
        "--nfe",
        type=int,
        default=64,
        help="Number of function evaluations",
    )
    parser.add_argument(
        "--parallel_mode",
        action="store_true",
        help="Shuffle the audio paths and skip the existing ones, enabling multiple jobs to run in parallel",
    )

    args = parser.parse_args()

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No CUDA devices available, running on CPU")
        device = "cpu"
        num_gpus = 1

    start_time = time.perf_counter()

    paths = sorted(args.in_dir.glob(f"**/*{args.suffix}"))
    if args.parallel_mode:
        random.shuffle(paths)

    if len(paths) == 0:
        print(f"No {args.suffix} files found in the following path: {args.in_dir}")
        return

    if num_gpus > 1:
        # Set start method before creating processes
        mp.set_start_method('spawn', force=True)

        processes = []
        for rank in range(num_gpus):
            p = mp.Process(
                target=process_files,
                args=(rank, num_gpus, args, paths)
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        process_files(0, 1, args, paths)

    elapsed_time = time.perf_counter() - start_time
    print(f"🌟 Enhancement done! {len(paths)} files processed in {elapsed_time:.2f}s")


if __name__ == "__main__":
    main()
