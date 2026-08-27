#!/usr/bin/env python3
"""
tools/dump_matmul_fixtures.py

Extracts real tensor fixtures for IQ4_XS matmul benchmarking and generates
activation vectors/matrices for canonical Qwen3.8-27B shapes.

For each canonical shape (K,N), extracts W_raw from GGUF (if available) and
generates activation inputs x (K) or X (K x M) with Gaussian N(0,1) seed,
plus y_ref computed via CPU dequant + GEMM.

Outputs to kernels/fixtures/matmul_<shape>_<K>x<N>_M*.npz and manifest.
"""

import argparse
import hashlib
import json
import os
import pathlib
import struct
import sys
from pathlib import Path

import numpy as np

try:
    import gguf
    from gguf.constants import GGMLQuantizationType
    from gguf.quants import dequantize
    HAS_GGUF = True
except Exception as e:
    HAS_GGUF = False
    print(f"[warn] gguf-py not available: {e}", file=sys.stderr)

# Canonical shapes: (name, K, N)
CANONICAL_SHAPES = [
    ("attn_q",    5120,  5120),
    ("attn_k",    5120,  5120),
    ("attn_v",    5120,  5120),
    ("attn_gate", 5120,  6144),
    ("attn_out",  5120,  5120),
    ("ffn_gate",  5120, 17408),
    ("ffn_up",    5120, 17408),
    ("ffn_down", 17408,  5120),
]

# Map canonical name -> list of candidate GGUF tensor names (first hit wins)
TENSOR_CANDIDATES = {
    "attn_q":    ["blk.0.attn_q.weight", "blk.3.attn_q.weight", "blk.0.attn_qkv.weight", "blk.0.attn_gate.weight"],
    "attn_k":    ["blk.3.attn_k.weight", "blk.0.attn_qkv.weight", "blk.0.attn_gate.weight"],
    "attn_v":    ["blk.3.attn_v.weight", "blk.0.attn_v.weight", "blk.0.attn_qkv.weight"],
    "attn_gate": ["blk.0.attn_gate.weight"],
    "attn_out":  ["blk.0.attn_output.weight", "blk.3.attn_output.weight", "blk.0.attn_gate.weight"],
    "ffn_gate":  ["blk.0.ffn_gate.weight"],
    "ffn_up":    ["blk.0.ffn_up.weight"],
    "ffn_down":  ["blk.0.ffn_down.weight"],
}

KVALUES_IQ4NL = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], dtype=np.int8)
QK_K = 256
BLOCK_BYTES = 136

def fp16_to_fp32_array(h):
    return h.astype(np.float16).astype(np.float32)

def extract_weight_for_shape(reader, shape_name, K, N):
    """Extract W raw blocks for shape K x N from GGUF reader."""
    if reader is None or not HAS_GGUF:
        return None, "no_reader"
    tensor_map = {t.name: t for t in reader.tensors}
    for cand in TENSOR_CANDIDATES.get(shape_name, []):
        if cand in tensor_map:
            t = tensor_map[cand]
            if t.tensor_type != GGMLQuantizationType.IQ4_XS:
                continue
            try:
                K_real = int(t.shape[0])
                N_real = int(t.shape[1])
                all_raw = t.data.reshape(-1, 136)
                blocks_per_row_real = K_real // QK_K
                blocks_per_row_canon = K // QK_K
                if K == K_real and N <= N_real:
                    need_blocks = N * blocks_per_row_canon
                    raw_slice = all_raw[:need_blocks].copy()
                    return raw_slice, cand
                elif N == N_real and K <= K_real and N <= 8192:
                    raw_2d = all_raw.reshape(N_real, blocks_per_row_real, 136)
                    sliced = raw_2d[:N, :blocks_per_row_canon, :].reshape(-1, 136).copy()
                    return sliced, cand
                elif K == K_real and N == N_real:
                    return all_raw.copy(), cand
                elif K <= K_real and N <= N_real:
                    raw_2d = all_raw.reshape(N_real, blocks_per_row_real, 136)
                    sliced = raw_2d[:N, :blocks_per_row_canon, :].reshape(-1, 136).copy()
                    return sliced, cand
            except Exception as e:
                print(f"[warn] extract {cand} failed: {e}", file=sys.stderr)
                continue
    return None, "no_match"

def synthetic_weight_blocks(K, N, seed=0):
    """Deterministic synthetic weight blocks mimicking GGUF entropy."""
    rng = np.random.default_rng(seed)
    blocks_per_row = K // QK_K
    total_blocks = N * blocks_per_row
    raw = np.zeros((total_blocks, 136), dtype=np.uint8)
    for b in range(total_blocks):
        d_val = float(rng.uniform(0.008, 0.8))
        d_bytes = np.float16(d_val).tobytes()
        raw[b, 0:2] = np.frombuffer(d_bytes, dtype=np.uint8)
        sh = int(rng.integers(0, 65536))
        raw[b, 2:4] = struct.pack("<H", sh)
        raw[b, 4:8] = rng.integers(0, 256, size=4, dtype=np.uint8)
        raw[b, 8:136] = rng.integers(0, 256, size=128, dtype=np.uint8)
    return raw

def cpu_gemv_reference(W_raw, x_vec, K, N):
    """CPU GEMV reference."""
    if HAS_GGUF:
        W_f32 = dequantize(W_raw, GGMLQuantizationType.IQ4_XS)
        W_mat = W_f32.reshape(N, K)
        y = W_mat @ x_vec
        return y.astype(np.float32)
    else:
        raise RuntimeError("gguf required")

def cpu_gemm_reference(W_raw, X_mat, K, N, M):
    """CPU GEMM reference: Y = W * X."""
    if HAS_GGUF:
        W_f32 = dequantize(W_raw, GGMLQuantizationType.IQ4_XS).reshape(N, K)
        Y = W_f32 @ X_mat
        return Y.astype(np.float32)
    else:
        raise RuntimeError("gguf required")

def main():
    parser = argparse.ArgumentParser(description="Dump IQ4_XS matmul fixtures for canonical shapes")
    parser.add_argument("--model", type=str, default="models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf", help="GGUF path")
    parser.add_argument("--out", type=str, default="kernels/fixtures", help="Output directory")
    parser.add_argument("--shapes", nargs="*", default=None, help="Subset of shape names to dump")
    parser.add_argument("--ms", nargs="*", type=int, default=[1, 16, 128, 512], help="M values to generate")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for activations")
    parser.add_argument("--manifest", type=str, default=None, help="Output manifest name")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model)
    reader = None
    if HAS_GGUF and model_path.exists():
        print(f"[dump_matmul] Opening GGUF {model_path}...")
        reader = gguf.GGUFReader(str(model_path))
        print(f"[dump_matmul] Found {len(reader.tensors)} tensors")
    else:
        print(f"[dump_matmul] No GGUF reader, using synthetic weights only")
        reader = None

    target_shapes = CANONICAL_SHAPES
    if args.shapes:
        sel = set(args.shapes)
        target_shapes = [s for s in CANONICAL_SHAPES if s[0] in sel]
        if not target_shapes:
            print(f"[error] no matching shapes for {args.shapes}", file=sys.stderr)
            sys.exit(1)

    rng_act = np.random.default_rng(args.seed)

    manifest_entries = []
    for (name, K, N) in target_shapes:
        print(f"\n[shape] {name}: K={K} N={N} blocks_per_row={K//QK_K}")
        W_raw, src_tensor = extract_weight_for_shape(reader, name, K, N)
        if W_raw is None:
            stable_seed = int(hashlib.sha256(name.encode("utf-8")).hexdigest()[:8], 16) ^ args.seed
            print(f"  -> No matching GGUF tensor for {name}, generating synthetic (seed={stable_seed})")
            W_raw = synthetic_weight_blocks(K, N, seed=stable_seed)
            src_tensor = "synthetic"
        else:
            print(f"  -> Extracted from {src_tensor}: W_raw shape {W_raw.shape}, bytes {W_raw.nbytes}")

        expected_blocks = N * (K // QK_K)
        assert W_raw.shape[0] == expected_blocks and W_raw.shape[1] == 136, f"W_raw shape mismatch {W_raw.shape} vs expected {(expected_blocks,136)}"

        W_sha = hashlib.sha256(W_raw.tobytes()).hexdigest()
        w_npz_path = out_dir / f"matmul_{name}_{K}x{N}_W.npz"
        w_bin_path = out_dir / f"matmul_{name}_{K}x{N}_W.bin"
        np.savez_compressed(w_npz_path, W_raw=W_raw, K=np.int64(K), N=np.int64(N), block_size=np.int64(136), QK_K=np.int64(QK_K), sha256=W_sha, source=src_tensor)
        W_raw.tofile(w_bin_path)
        print(f"  -> W saved {w_npz_path.name} sha256 {W_sha[:12]}...")

        for M in args.ms:
            X = rng_act.standard_normal(size=(K, M)).astype(np.float32)
            print(f"  [M={M}] generating X {X.shape} and Y_ref...")
            if M == 1:
                x_vec = X[:, 0]
                y_ref = cpu_gemv_reference(W_raw, x_vec, K, N)
                npz_path = out_dir / f"matmul_{name}_{K}x{N}_M{M}.npz"
                np.savez_compressed(npz_path, W_raw=W_raw, x=x_vec, X=X, y_ref=y_ref, K=np.int64(K), N=np.int64(N), M=np.int64(M), source=src_tensor, W_sha256=W_sha)
                y_bin_path = out_dir / f"matmul_{name}_{K}x{N}_M{M}_y_ref.bin"
                y_ref.tofile(y_bin_path)
                x_bin_path = out_dir / f"matmul_{name}_{K}x{N}_M{M}_x.bin"
                x_vec.tofile(x_bin_path)
                manifest_entries.append({
                    "name": name,
                    "K": int(K),
                    "N": int(N),
                    "M": int(M),
                    "W_shape": [int(N), int(K)],
                    "W_blocks": int(expected_blocks),
                    "W_source": src_tensor,
                    "W_sha256": W_sha,
                    "W_npz": w_npz_path.name,
                    "W_bin": w_bin_path.name,
                    "X_shape": [int(K), int(M)],
                    "Y_shape": [int(N), int(M)],
                    "npz": npz_path.name,
                    "x_bin": x_bin_path.name,
                    "y_ref_bin": y_bin_path.name,
                    "dtype": "IQ4_XS",
                    "seed": int(args.seed),
                })
            else:
                y_ref = cpu_gemm_reference(W_raw, X, K, N, M)
                npz_path = out_dir / f"matmul_{name}_{K}x{N}_M{M}.npz"
                np.savez_compressed(npz_path, W_raw=W_raw, X=X, Y_ref=y_ref, K=np.int64(K), N=np.int64(N), M=np.int64(M), source=src_tensor, W_sha256=W_sha)
                Y_bin_path = out_dir / f"matmul_{name}_{K}x{N}_M{M}_Y_ref.bin"
                y_ref.tofile(Y_bin_path)
                X_bin_path = out_dir / f"matmul_{name}_{K}x{N}_M{M}_X.bin"
                X.tofile(X_bin_path)
                manifest_entries.append({
                    "name": name,
                    "K": int(K),
                    "N": int(N),
                    "M": int(M),
                    "W_shape": [int(N), int(K)],
                    "W_blocks": int(expected_blocks),
                    "W_source": src_tensor,
                    "W_sha256": W_sha,
                    "W_npz": w_npz_path.name,
                    "W_bin": w_bin_path.name,
                    "X_shape": [int(K), int(M)],
                    "Y_shape": [int(N), int(M)],
                    "npz": npz_path.name,
                    "X_bin": X_bin_path.name,
                    "Y_ref_bin": Y_bin_path.name,
                    "dtype": "IQ4_XS",
                    "seed": int(args.seed),
                })
            print(f"     -> saved {npz_path.name}")

    manifest_path = out_dir / (args.manifest or "manifest_matmul.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_entries, f, indent=2)
    print(f"\n[done] Wrote {len(manifest_entries)} entries to {manifest_path}")

if __name__ == "__main__":
    main()
