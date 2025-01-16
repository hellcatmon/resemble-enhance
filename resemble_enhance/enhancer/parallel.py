import torch
import torchaudio
from tqdm import tqdm
from pathlib import Path
from .inference import denoise, enhance


def process_files(rank, world_size, args, file_paths):
    device = f'cuda:{rank}'
    torch.cuda.set_device(device)

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
