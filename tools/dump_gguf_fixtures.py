#!/usr/bin/env python3
"""
tools/dump_gguf_fixtures.py

Extracts real tensor fixtures from GGUF models and creates synthetic test fixtures
for IQ4_XS (136 bytes per 256 weights) kernel development and testing.

Zero llama.cpp runtime dependency — uses gguf-py and numpy.
"""

import argparse
import hashlib
import json
import os
import pathlib
import struct
import numpy as np
import gguf
from gguf.constants import GGMLQuantizationType, GGML_QUANT_SIZES
from gguf.quants import dequantize

KVALUES_IQ4NL = np.array(
    [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
    dtype=np.int8
)

def fp32_to_fp16_bytes(val: float) -> bytes:
    """Converts a float to 2-byte IEEE 754 half-precision float in little-endian."""
    return np.float16(val).tobytes()

def fp16_bytes_to_fp32(b: bytes) -> float:
    """Converts 2-byte IEEE 754 half-precision float to float32."""
    return float(np.frombuffer(b, dtype=np.float16)[0])

def make_synthetic_block(
    d_val: float,
    scales_h_val: int,
    scales_l_bytes: bytes,
    qs_bytes: bytes
) -> bytes:
    """Constructs a single 136-byte block_iq4_xs."""
    assert len(scales_l_bytes) == 4
    assert len(qs_bytes) == 128
    d_bytes = fp32_to_fp16_bytes(d_val)
    sh_bytes = struct.pack("<H", scales_h_val & 0xFFFF)
    block = d_bytes + sh_bytes + scales_l_bytes + qs_bytes
    assert len(block) == 136
    return block

def generate_synthetic_fixtures() -> dict[str, dict]:
    """Generates deterministic synthetic fixtures covering corner cases."""
    fixtures = {}

    # Case 1: Zero block (d=0.0, scales=0, qs=0)
    b_zero = make_synthetic_block(0.0, 0, bytes(4), bytes(128))
    fixtures["synthetic_zero"] = {
        "raw": np.frombuffer(b_zero, dtype=np.uint8).reshape(1, 136),
        "case": "zero_block",
        "desc": "All fields zeroed out"
    }

    # Case 2: Max scale (+31) and Min scale (-32)
    # ls = low | (high << 4). Max ls = 15 | (3 << 4) = 63 -> (63 - 32) = +31.
    # Min ls = 0 | (0 << 4) = 0 -> (0 - 32) = -32.
    b_max_scale = make_synthetic_block(
        d_val=0.5,
        scales_h_val=0xFFFF, # high 2 bits = 3 for all 8 sub-blocks
        scales_l_bytes=bytes([0xFF] * 4), # low 4 bits = 15 for all 8 sub-blocks
        qs_bytes=bytes([(i & 0x0F) | (((15 - i) & 0x0F) << 4) for i in range(128)])
    )
    b_min_scale = make_synthetic_block(
        d_val=0.5,
        scales_h_val=0x0000, # high 2 bits = 0
        scales_l_bytes=bytes([0x00] * 4), # low 4 bits = 0
        qs_bytes=bytes([(i & 0x0F) | (((15 - i) & 0x0F) << 4) for i in range(128)])
    )
    fixtures["synthetic_max_scale"] = {
        "raw": np.concatenate([
            np.frombuffer(b_max_scale, dtype=np.uint8).reshape(1, 136),
            np.frombuffer(b_min_scale, dtype=np.uint8).reshape(1, 136)
        ], axis=0),
        "case": "max_min_scale",
        "desc": "Max scale ls=+31 and Min scale ls=-32 blocks"
    }

    # Case 3: Split-half boundary (stresses low nibble @ index j vs high nibble @ index j+16)
    # For sub-block s=0: qs[0]=0x12 -> low nibble = 2 (index 0), high nibble = 1 (index 16)
    qs_split = bytearray(128)
    for s in range(8):
        for j in range(16):
            lo_idx = (s * 2 + 1) % 16
            hi_idx = (s * 2 + 2) % 16
            qs_split[s * 16 + j] = (lo_idx & 0x0F) | ((hi_idx & 0x0F) << 4)

    # Alternate scales across sub-blocks
    scales_l_split = bytes([0x21, 0x43, 0x65, 0x87]) # low scales
    scales_h_split = 0b1110010011100100 # 2 bits per sub-block: 0, 1, 2, 3, 0, 1, 2, 3
    b_split = make_synthetic_block(1.25, scales_h_split, scales_l_split, bytes(qs_split))
    fixtures["synthetic_split_half"] = {
        "raw": np.frombuffer(b_split, dtype=np.uint8).reshape(1, 136),
        "case": "split_half_boundary",
        "desc": "Split-half layout lo@j / hi@j+16 with varied sub-block scales"
    }

    # Case 4: Nibble extremes (all 0, all 15, alternating 0x0F / 0xF0)
    b_nib_0 = make_synthetic_block(1.0, 0x5555, bytes([0x33] * 4), bytes([0x00] * 128))
    b_nib_f = make_synthetic_block(1.0, 0x5555, bytes([0x33] * 4), bytes([0xFF] * 128))
    b_nib_alt = make_synthetic_block(1.0, 0x5555, bytes([0x33] * 4), bytes([0x0F if i % 2 == 0 else 0xF0 for i in range(128)]))
    fixtures["synthetic_nibble_extremes"] = {
        "raw": np.concatenate([
            np.frombuffer(b_nib_0, dtype=np.uint8).reshape(1, 136),
            np.frombuffer(b_nib_f, dtype=np.uint8).reshape(1, 136),
            np.frombuffer(b_nib_alt, dtype=np.uint8).reshape(1, 136)
        ], axis=0),
        "case": "nibble_extremes",
        "desc": "All zeros, all 15s, and alternating 0x0F/0xF0 nibbles"
    }

    # Case 5: Sub-block isolated (only sub-block 3 non-zero, rest zero)
    # sub-block 3 is in scales_l[1] high nibble (ib=3 -> ib/2=1, 4*(3%2)=4)
    # scales_h sub-block 3 is bits [7:6]
    scales_l_iso = bytearray(4)
    scales_l_iso[1] = 0x50 # low nibble (sub-block 2)=0, high nibble (sub-block 3)=5
    scales_h_iso = 0b0000000010000000 # sub-block 3 high bits = 2
    qs_iso = bytearray(128)
    for j in range(16):
        qs_iso[3 * 16 + j] = 0x79 # non-zero nibbles only in sub-block 3

    b_iso = make_synthetic_block(2.0, scales_h_iso, bytes(scales_l_iso), bytes(qs_iso))
    fixtures["synthetic_subblock_isolated"] = {
        "raw": np.frombuffer(b_iso, dtype=np.uint8).reshape(1, 136),
        "case": "subblock_isolated",
        "desc": "Only sub-block 3 has non-zero scale and weights, rest 0"
    }

    return fixtures

def dump_fixtures(
    model_path: str | None,
    tensors: list[str],
    num_blocks: int,
    out_dir: pathlib.Path,
    generate_synthetic: bool
):
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_entries = []

    # 1. Process synthetic fixtures if requested
    if generate_synthetic:
        syn_fixtures = generate_synthetic_fixtures()
        for name, data in syn_fixtures.items():
            raw = data["raw"] # shape (N, 136)
            f32 = dequantize(raw, GGMLQuantizationType.IQ4_XS) # shape (N, 256)
            n_blocks = raw.shape[0]

            npz_path = out_dir / f"{name}.iq4xs.npz"
            bin_path = out_dir / f"{name}.bin"
            f32_bin_path = out_dir / f"{name}.f32.bin"

            np.savez(npz_path, raw=raw, f32=f32, n_blocks=n_blocks, block_size=136, QK_K=256)
            raw.tofile(bin_path)
            f32.tofile(f32_bin_path)

            raw_sha256 = hashlib.sha256(raw.tobytes()).hexdigest()

            entry = {
                "name": name,
                "tensor_type": "IQ4_XS",
                "synthetic": True,
                "case": data["case"],
                "description": data["desc"],
                "n_blocks": n_blocks,
                "block_size": 136,
                "QK_K": 256,
                "raw_sha256": raw_sha256,
                "commit": "bb4caa75",
                "rocm": "7.2.1",
                "path": npz_path.name,
                "bin_path": bin_path.name,
                "f32_bin_path": f32_bin_path.name
            }
            manifest_entries.append(entry)
            print(f"[Fixture] Generated synthetic fixture {npz_path.name}: {n_blocks} blocks, sha256={raw_sha256[:12]}...")

    # 2. Process real GGUF model tensors
    if model_path and os.path.exists(model_path):
        print(f"[GGUF] Opening model {model_path}...")
        reader = gguf.GGUFReader(model_path)
        tensor_map = {t.name: t for t in reader.tensors}

        for tensor_name in tensors:
            if tensor_name not in tensor_map:
                print(f"[GGUF Warning] Tensor '{tensor_name}' not found in model.")
                continue

            t = tensor_map[tensor_name]
            if t.tensor_type != GGMLQuantizationType.IQ4_XS:
                print(f"[GGUF Warning] Tensor '{tensor_name}' type is {t.tensor_type.name}, not IQ4_XS.")
                continue

            all_raw = t.data.reshape(-1, 136)
            n_avail = all_raw.shape[0]
            take_blocks = min(num_blocks, n_avail)
            raw = np.ascontiguousarray(all_raw[:take_blocks])
            f32 = dequantize(raw, GGMLQuantizationType.IQ4_XS)

            safe_name = tensor_name.replace(".", "_")
            npz_path = out_dir / f"{safe_name}.iq4xs.npz"
            bin_path = out_dir / f"{safe_name}.bin"
            f32_bin_path = out_dir / f"{safe_name}.f32.bin"

            np.savez(npz_path, raw=raw, f32=f32, n_blocks=take_blocks, block_size=136, QK_K=256, orig_shape=np.array(t.shape))
            raw.tofile(bin_path)
            f32.tofile(f32_bin_path)

            raw_sha256 = hashlib.sha256(raw.tobytes()).hexdigest()

            entry = {
                "name": tensor_name,
                "tensor_type": "IQ4_XS",
                "synthetic": False,
                "shape": [int(x) for x in t.shape],
                "n_blocks": int(take_blocks),
                "total_tensor_blocks": int(n_avail),
                "block_size": 136,
                "QK_K": 256,
                "raw_sha256": raw_sha256,
                "commit": "bb4caa75",
                "rocm": "7.2.1",
                "path": npz_path.name,
                "bin_path": bin_path.name,
                "f32_bin_path": f32_bin_path.name
            }
            manifest_entries.append(entry)
            print(f"[GGUF] Dumped {tensor_name} -> {npz_path.name}: {take_blocks} blocks, sha256={raw_sha256[:12]}...")

    # Write manifest
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_entries, f, indent=2)
    print(f"[Manifest] Wrote {len(manifest_entries)} entries to {manifest_path}")

def main():
    parser = argparse.ArgumentParser(description="Dump IQ4_XS GGUF and synthetic fixtures")
    parser.add_argument("--model", type=str, default="models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf", help="Path to GGUF model")
    parser.add_argument("--tensors", nargs="+", default=["blk.0.ffn_down.weight", "blk.0.attn_gate.weight", "token_embd.weight"], help="Tensors to dump")
    parser.add_argument("--num-blocks", type=int, default=8, help="Number of 136-byte blocks to dump per tensor")
    parser.add_argument("--out", type=str, default="kernels/fixtures", help="Output directory")
    parser.add_argument("--synthetic", action="store_true", default=True, help="Generate synthetic test fixtures")
    args = parser.parse_args()

    dump_fixtures(
        model_path=args.model if os.path.exists(args.model) else None,
        tensors=args.tensors,
        num_blocks=args.num_blocks,
        out_dir=pathlib.Path(args.out),
        generate_synthetic=args.synthetic
    )

if __name__ == "__main__":
    main()
