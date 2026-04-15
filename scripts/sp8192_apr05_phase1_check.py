#!/usr/bin/env python3
"""Phase 1 readiness checks for the April 5 SP8192 reproduction target."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
APR05_RECORD = (
    REPO_ROOT
    / "records"
    / "track_10min_16mb"
    / "2026-04-05_SP8192_GPTQ-Embeddings_SDClip_Loop45x2"
)
APR05_HUMAN_SCRIPT = APR05_RECORD / "train_gpt_human.py"


def import_or_report(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        print(f"missing {module_name}: {exc.__class__.__name__}: {exc}")
        return False
    print(f"ok {module_name}")
    return True


def load_hf_manifest() -> dict:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="kevclark/parameter-golf",
        filename="manifest.json",
        subfolder="datasets",
        repo_type="dataset",
    )
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_manifest() -> None:
    manifest = load_hf_manifest()
    dataset = next(
        (entry for entry in manifest.get("datasets", []) if entry.get("name") == "fineweb10B_sp8192"),
        None,
    )
    tokenizer = next(
        (entry for entry in manifest.get("tokenizers", []) if entry.get("name") == "sp_bpe_8192"),
        None,
    )
    if dataset is None:
        raise RuntimeError("fineweb10B_sp8192 is missing from kevclark/parameter-golf manifest")
    if tokenizer is None:
        raise RuntimeError("sp_bpe_8192 tokenizer is missing from kevclark/parameter-golf manifest")
    stats = dataset.get("stats") or {}
    print(
        "ok hf manifest: "
        f"train_shards={stats.get('files_train')} "
        f"val_shards={stats.get('files_val')} "
        f"tokenizer={tokenizer.get('model_path')}"
    )


def remove_stale_manifest_if_needed() -> None:
    manifest_path = DATA_DIR / "manifest.json"
    if not manifest_path.exists():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        manifest_path.unlink()
        return
    has_sp8192 = any(
        entry.get("name") == "fineweb10B_sp8192" for entry in manifest.get("datasets", [])
    )
    if not has_sp8192:
        print("removing stale data/manifest.json without sp8192 entries")
        manifest_path.unlink()


def download_data(train_shards: int) -> None:
    remove_stale_manifest_if_needed()
    env = os.environ.copy()
    env["MATCHED_FINEWEB_REPO_ID"] = "kevclark/parameter-golf"
    cmd = [
        sys.executable,
        "data/cached_challenge_fineweb.py",
        "--variant",
        "sp8192",
        "--train-shards",
        str(train_shards),
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)
    print(f"ok downloaded sp8192 data with train_shards={train_shards}")


def check_local_data(require: bool) -> bool:
    tokenizer = DATA_DIR / "tokenizers" / "fineweb_8192_bpe.model"
    train_files = sorted((DATA_DIR / "datasets" / "fineweb10B_sp8192").glob("fineweb_train_*.bin"))
    val_files = sorted((DATA_DIR / "datasets" / "fineweb10B_sp8192").glob("fineweb_val_*.bin"))
    present = tokenizer.exists() and bool(train_files) and bool(val_files)
    print(
        "local sp8192 data: "
        f"tokenizer={'yes' if tokenizer.exists() else 'no'} "
        f"train_shards={len(train_files)} "
        f"val_shards={len(val_files)}"
    )
    if require and not present:
        raise FileNotFoundError(
            "local sp8192 data is incomplete; rerun with --download-data or download it on the pod"
        )
    if tokenizer.exists():
        import sentencepiece as spm

        sp = spm.SentencePieceProcessor(model_file=str(tokenizer))
        if int(sp.vocab_size()) != 8192:
            raise RuntimeError(f"expected sp8192 tokenizer, got vocab_size={sp.vocab_size()}")
        print("ok local tokenizer vocab_size=8192")
    return present


def configure_smoke_environment() -> None:
    defaults = {
        "DATA_DIR": str(DATA_DIR),
        "VOCAB_SIZE": "8192",
        "TRAIN_SEQ_LEN": "128",
        "ROPE_TRAIN_SEQ_LEN": "128",
        "EVAL_SEQ_LEN": "128",
        "TRAIN_BATCH_TOKENS": "1024",
        "VAL_BATCH_TOKENS": "1024",
        "WARMUP_STEPS": "0",
        "ITERATIONS": "1",
        "VAL_LOSS_EVERY": "0",
        "MAX_WALLCLOCK_SECONDS": "0",
        "GPTQ_CALIBRATION_BATCHES": "1",
        "RUN_ID": "phase1_sp8192_apr05_check",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def load_apr05_module():
    configure_smoke_environment()
    spec = importlib.util.spec_from_file_location("apr05_sp8192_train_gpt_human", APR05_HUMAN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {APR05_HUMAN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cuda_forward(use_loader: bool, train_step: bool) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for CUDA smoke checks")
    module = load_apr05_module()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    h = module.Hyperparameters()
    h.distributed = False
    h.rank = 0
    h.world_size = 1
    h.local_rank = 0
    h.is_main_process = True
    h.grad_accum_steps = 8

    model = module.GPT(h).to(device).bfloat16()
    module.restore_fp32_params(model)
    model.train(mode=train_step)
    if use_loader:
        loader = module.ShuffledSequenceLoader(h, device)
        x, y = loader.next_batch(h.train_batch_tokens, h.grad_accum_steps)
    else:
        x = torch.randint(0, h.vocab_size, (1, h.train_seq_len), device=device, dtype=torch.long)
        y = torch.randint(0, h.vocab_size, (1, h.train_seq_len), device=device, dtype=torch.long)
    if train_step:
        optimizers = module.Optimizers(h, model)
        optimizers.zero_grad_all()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(x, y)
        loss.backward()
        optimizers.step()
        label = "train step"
    else:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(x, y)
        label = "forward"
    torch.cuda.synchronize()
    print(f"ok cuda {label}: loss={float(loss.detach()):.4f} loader={'yes' if use_loader else 'no'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-data", action="store_true", help="Download sp8192 val plus train shards.")
    parser.add_argument("--train-shards", type=int, default=1, help="Training shard count for --download-data.")
    parser.add_argument("--require-local-data", action="store_true", help="Fail if local sp8192 files are missing.")
    parser.add_argument("--cuda-forward", action="store_true", help="Run a tiny synthetic CUDA forward pass.")
    parser.add_argument(
        "--loader-forward",
        action="store_true",
        help="Run the CUDA forward pass using ShuffledSequenceLoader; requires local data.",
    )
    parser.add_argument(
        "--train-step",
        action="store_true",
        help="Run one CUDA backward/optimizer step using ShuffledSequenceLoader; requires local data.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    required_modules = ["numpy", "sentencepiece", "huggingface_hub"]
    if args.cuda_forward or args.loader_forward or args.train_step:
        required_modules.extend(["torch", "brotli", "flash_attn_interface"])
    ok = all(import_or_report(name) for name in required_modules)
    if not ok:
        raise SystemExit(1)
    check_manifest()
    if args.download_data:
        download_data(args.train_shards)
    check_local_data(require=args.require_local_data or args.loader_forward or args.train_step)
    if args.cuda_forward or args.loader_forward or args.train_step:
        cuda_forward(use_loader=args.loader_forward or args.train_step, train_step=args.train_step)


if __name__ == "__main__":
    main()
