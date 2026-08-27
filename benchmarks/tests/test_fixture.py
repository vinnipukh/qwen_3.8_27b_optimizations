import json
import hashlib
import pathlib
import numpy as np
import pytest

KVALUES_IQ4NL = np.array(
    [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
    dtype=np.int8
)

def dequantize_iq4xs_numpy(raw_bytes: bytes) -> np.ndarray:
    """Pure Python / NumPy independent reference dequantizer for block_iq4_xs."""
    assert len(raw_bytes) % 136 == 0
    n_blocks = len(raw_bytes) // 136
    out = np.zeros((n_blocks, 256), dtype=np.float32)

    for b in range(n_blocks):
        blk_bytes = raw_bytes[b * 136 : (b + 1) * 136]
        d_fp16 = np.frombuffer(blk_bytes[:2], dtype=np.float16)[0]
        d = float(d_fp16)

        scales_h = int(np.frombuffer(blk_bytes[2:4], dtype=np.uint16)[0])
        scales_l = np.frombuffer(blk_bytes[4:8], dtype=np.uint8)
        qs = np.frombuffer(blk_bytes[8:136], dtype=np.uint8)

        for s in range(8):
            ls_low = (int(scales_l[s // 2]) >> (4 * (s & 1))) & 0x0F
            ls_high = (scales_h >> (2 * s)) & 0x03
            ls = ls_low | (ls_high << 4)
            scale = d * (ls - 32)

            for j in range(16):
                byte_val = int(qs[s * 16 + j])
                lo_nib = byte_val & 0x0F
                hi_nib = (byte_val >> 4) & 0x0F

                out[b, s * 32 + j] = scale * float(KVALUES_IQ4NL[lo_nib])
                out[b, s * 32 + j + 16] = scale * float(KVALUES_IQ4NL[hi_nib])

    return out

def test_manifest_and_fixtures_integrity():
    fixtures_dir = pathlib.Path("kernels/fixtures")
    manifest_path = fixtures_dir / "manifest.json"
    assert manifest_path.exists(), "manifest.json must exist in kernels/fixtures"
    manifest_dequant_path = fixtures_dir / "manifest_dequant.json"
    assert manifest_dequant_path.exists(), "manifest_dequant.json must exist in kernels/fixtures"
    manifest_matmul_path = fixtures_dir / "manifest_matmul.json"
    assert manifest_matmul_path.exists(), "manifest_matmul.json must exist in kernels/fixtures"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert len(manifest) >= 8, f"Expected at least 8 fixtures, found {len(manifest)}"

    for entry in manifest:
        npz_file = fixtures_dir / entry["path"]
        bin_file = fixtures_dir / entry["bin_path"]
        f32_bin_file = fixtures_dir / entry["f32_bin_path"]

        assert npz_file.exists(), f"Missing {npz_file}"
        assert bin_file.exists(), f"Missing {bin_file}"
        assert f32_bin_file.exists(), f"Missing {f32_bin_file}"

        data = np.load(npz_file)
        raw = data["raw"]
        f32 = data["f32"]

        assert raw.shape[-1] == 136, "Block size must be 136 bytes"
        assert f32.shape[-1] == 256, "Dequantized elements per block must be 256"

        computed_sha = hashlib.sha256(raw.tobytes()).hexdigest()
        assert computed_sha == entry["raw_sha256"], f"SHA256 mismatch for {entry['name']}"

        # Test independent dequantization matches stored f32
        f32_ref = dequantize_iq4xs_numpy(raw.tobytes())
        diff = np.abs(f32 - f32_ref)
        max_diff = float(np.max(diff))
        assert max_diff < 1e-5, f"Dequant mismatch {max_diff} on {entry['name']}"

def test_synthetic_split_half_layout():
    fixtures_dir = pathlib.Path("kernels/fixtures")
    npz_file = fixtures_dir / "synthetic_split_half.iq4xs.npz"
    data = np.load(npz_file)
    raw = data["raw"]
    f32 = data["f32"]

    # Verify split-half property: low nibble at j=0 vs high nibble at j=16
    assert f32.shape == (1, 256)
    # Both positions must be non-zero and properly decoded
    assert f32[0, 0] != 0.0
    assert f32[0, 16] != 0.0

def test_synthetic_zero_block():
    fixtures_dir = pathlib.Path("kernels/fixtures")
    npz_file = fixtures_dir / "synthetic_zero.iq4xs.npz"
    data = np.load(npz_file)
    f32 = data["f32"]
    assert np.all(f32 == 0.0), "Zero block must dequantize to all exact zeros"
