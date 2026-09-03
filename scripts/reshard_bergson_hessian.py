#!/usr/bin/env python3
"""Repartition a Bergson factored Hessian without recomputing it."""

from __future__ import annotations

import argparse
import gc
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from bergson.hessians.sharded_computation import shard_bounds


ROW_SHARDED = (
    "activation_sharded",
    "eigen_activation_sharded",
    "eigen_gradient_sharded",
    "eigenvalue_correction_sharded",
    "eigenvalue_sharded",
    "factor_eig_g",
    "gradient_sharded",
)
REPLICATED = ("factor_eig_a",)


def shard_paths(directory: Path) -> list[Path]:
    return sorted(directory.glob("shard_*.safetensors"), key=lambda p: int(p.stem[6:]))


def reshard_directory(source: Path, destination: Path, world_size: int) -> None:
    inputs = shard_paths(source)
    if not inputs:
        raise FileNotFoundError(f"no shards in {source}")
    loaded = [load_file(str(path), device="cpu") for path in inputs]
    keys = tuple(loaded[0])
    if any(tuple(shard) != keys for shard in loaded[1:]):
        raise ValueError(f"inconsistent keys in {source}")

    full = {key: torch.cat([shard[key] for shard in loaded], dim=0) for key in keys}
    destination.mkdir(parents=True)
    for rank in range(world_size):
        output = {}
        for key, tensor in full.items():
            start, stop = shard_bounds(tensor.shape[0], rank, world_size)
            output[key] = tensor[start:stop].contiguous()
        save_file(output, str(destination / f"shard_{rank}.safetensors"))

    # Verify that concatenating the new partition exactly recovers every tensor.
    outputs = [load_file(str(path), device="cpu") for path in shard_paths(destination)]
    for key, expected in full.items():
        actual = torch.cat([shard[key] for shard in outputs], dim=0)
        if not torch.equal(actual, expected):
            raise ValueError(f"verification failed for {source.name}/{key}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--world-size", type=int, default=8)
    args = parser.parse_args()

    source = args.source.resolve()
    destination = args.destination
    marker = destination / "RESHARD_COMPLETE"
    if marker.exists():
        print(f"already complete: {destination}")
        return
    if destination.exists():
        shutil.rmtree(destination)

    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    source_kfac = source / "kfac"
    output_kfac = temporary / "kfac"
    output_kfac.mkdir(parents=True)

    for name in ROW_SHARDED:
        print(f"resharding {name}", flush=True)
        reshard_directory(source_kfac / name, output_kfac / name, args.world_size)
        gc.collect()

    for name in REPLICATED:
        source_shard = shard_paths(source_kfac / name)[0]
        output_dir = output_kfac / name
        output_dir.mkdir(parents=True)
        for rank in range(args.world_size):
            os.symlink(source_shard, output_dir / f"shard_{rank}.safetensors")

    os.symlink(source_kfac / "total_processed.pt", output_kfac / "total_processed.pt")
    (temporary / "RESHARD_COMPLETE").write_text(
        f"source={source}\nworld_size={args.world_size}\n"
    )
    temporary.rename(destination)
    print(f"complete: {destination}")


if __name__ == "__main__":
    main()
