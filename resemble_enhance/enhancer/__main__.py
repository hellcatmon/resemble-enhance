import argparse
import random
import time
import torch.multiprocessing as mp
from pathlib import Path
import torch
import torchaudio
from tqdm import tqdm
import os

from .inference import denoise, enhance


def process_files(rank, world_size, args, file_paths):
    # Set device for this process
    device = f'cuda:{rank}'
    torch.cuda.set_device(device)

    # Calculate chunk of files for this GPU
    chunk_size = len(file_paths) // world_size
    start_idx = rank * chunk_size
    end_idx = start_idx + chunk_size if rank < world_size - \
        1 else len(file_paths)
    gpu_files = file_paths[start_idx:end_idx]

    pbar = tqdm(gpu_files, position=rank, desc=f'GPU {rank}')

    for path in pbar:
        out_path = args.out_dir / path.relative_to(args.in_dir)
        if args.parallel_mode and out_path.exists():
            continue

        try:
            dwav, sr = torchaudio.load(path)
            dwav = dwav.mean(0)

            if args.denoise_only:
                hwav, sr = denoise(
                    dwav=dwav,
                    sr=sr,
                    device=device,
                    run_dir=args.run_dir,
                )
            else:
                hwav, sr = enhance(
                    dwav=dwav,
                    sr=sr,
                    device=device,
                    nfe=args.nfe,
                    solver=args.solver,
                    lambd=args.lambd,
                    tau=args.tau,
                    run_dir=args.run_dir,
                )

            del dwav
            torch.cuda.empty_cache()

            out_path.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(out_path, hwav[None], sr)

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\nGPU {rank}: Skipping {path} - file too large for available memory")
                torch.cuda.empty_cache()
                continue
            raise e


@torch.inference_mode()
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

    # Get number of available GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No CUDA devices available, running on CPU")
        device = "cpu"
        num_gpus = 1

    start_time = time.perf_counter()

    # Get all file paths
    paths = sorted(args.in_dir.glob(f"**/*{args.suffix}"))
    if args.parallel_mode:
        random.shuffle(paths)

    if len(paths) == 0:
        print(f"No {args.suffix} files found in the following path: {args.in_dir}")
        return

    if num_gpus > 1:
        # Multiprocessing for multiple GPUs
        mp.spawn(
            process_files,
            args=(num_gpus, args, paths),
            nprocs=num_gpus,
            join=True
        )
    else:
        # Single GPU/CPU processing
        process_files(0, 1, args, paths)

    elapsed_time = time.perf_counter() - start_time
    print(f"🌟 Enhancement done! {len(paths)} files processed in {elapsed_time:.2f}s")


if __name__ == "__main__":
    # Required for Windows support
    mp.set_start_method('spawn', force=True)
    main()
