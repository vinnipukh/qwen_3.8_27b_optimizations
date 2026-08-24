"""Unit tests for llabench wrapper module."""
import pytest
from benchmarks.lib import llabench
from benchmarks.tests.fixtures.gen_llabench_jsonl import (
    make_jsonl_row,
    make_tier_rows,
    write_jsonl_file,
)


def test_build_argv_golden_structure():
    argv = llabench.build_argv(
        prompt_tokens=4096,
        gen_pairs=[(4096, 128)],
        fa_seq=("off", "on"),
        repeats=5,
        delay_s=30,
        threads=8,
        bin_path="/custom/path/llama-bench",
        model="/custom/path/model.gguf",
    )

    # Must start with binary path
    assert argv[0] == "/custom/path/llama-bench"

    # Explicit -n 0 must be present to zero default gen vector
    assert "-n" in argv
    idx_n = argv.index("-n")
    assert argv[idx_n + 1] == "0"

    # -p must come before -pg
    assert "-p" in argv
    idx_p = argv.index("-p")
    assert argv[idx_p + 1] == "4096"

    assert "-pg" in argv
    idx_pg = argv.index("-pg")
    assert argv[idx_pg + 1] == "4096,128"
    assert idx_p < idx_pg

    # --delay must be used, NEVER -D
    assert "--delay" in argv
    assert "-D" not in argv
    idx_delay = argv.index("--delay")
    assert argv[idx_delay + 1] == "30"

    # fa_seq flag
    assert "-fa" in argv
    idx_fa = argv.index("-fa")
    assert argv[idx_fa + 1] == "off,on"

    # Output jsonl flags
    assert "-o" in argv and argv[argv.index("-o") + 1] == "jsonl"
    assert "-oe" in argv and argv[argv.index("-oe") + 1] == "jsonl"


def test_parse_rows_and_tier_integrity(tmp_path):
    jsonl_file = tmp_path / "tier4096.jsonl"
    clean_rows = make_tier_rows(tier=4096, gen=128, fa_seq=(0, 1))
    write_jsonl_file(str(jsonl_file), clean_rows)

    parsed = llabench.parse_rows(jsonl_file)
    assert len(parsed) == 4

    plans = llabench.enumerate_tiers(tiers=(4096,))
    assert len(plans) == 1
    llabench.assert_tier_rows(parsed, plans[0])


def test_contamination_detection_default_512():
    # Contaminated row where upstream default 512 prompt slipped into generation row
    bad_rows = [
        make_jsonl_row(n_prompt=4096, n_gen=0, flash_attn=0),
        make_jsonl_row(n_prompt=512, n_gen=128, flash_attn=0),  # Contaminated
    ]
    with pytest.raises(llabench.MatrixContaminationError, match="Contaminated by upstream default"):
        llabench.assert_cell_integrity(
            bad_rows,
            [
                {"type": "pp", "n_prompt": 4096, "n_gen": 0, "flash_attn": 0},
                {"type": "tg", "n_prompt": 4096, "n_gen": 128, "flash_attn": 0},
            ],
        )


def test_contamination_detection_empty_context():
    # Contaminated row with n_prompt=0 and n_gen=128
    bad_rows = [
        make_jsonl_row(n_prompt=0, n_gen=128, flash_attn=0),
    ]
    with pytest.raises(llabench.MatrixContaminationError, match="Banned empty-context"):
        llabench.assert_cell_integrity(
            bad_rows,
            [{"type": "tg", "n_prompt": 4096, "n_gen": 128, "flash_attn": 0}],
        )


def test_tier_enumeration_order():
    tiers = (16384, 4096, 32768, 8192)
    plans = llabench.enumerate_tiers(tiers=tiers, fa_seq=("off", "on"))

    # Must be sorted ascending
    assert [p.tier for p in plans] == [4096, 8192, 16384, 32768]

    # Each tier must have exactly 4 expected cells (pp off, tg off, pp on, tg on)
    for p in plans:
        assert len(p.expected_cells) == 4
        assert p.expected_cells[0] == {"type": "pp", "n_prompt": p.tier, "n_gen": 0, "flash_attn": 0}
        assert p.expected_cells[1] == {"type": "tg", "n_prompt": p.tier, "n_gen": 128, "flash_attn": 0}
        assert p.expected_cells[2] == {"type": "pp", "n_prompt": p.tier, "n_gen": 0, "flash_attn": 1}
        assert p.expected_cells[3] == {"type": "tg", "n_prompt": p.tier, "n_gen": 128, "flash_attn": 1}


def test_corpus_integrity():
    from pathlib import Path
    import hashlib

    corpus_dir = Path("benchmarks/prompts")
    txt_files = sorted(corpus_dir.glob("*.txt"))
    assert len(txt_files) == 6, f"Expected 6 .txt corpus files, found {len(txt_files)}"

    expected_files = {
        "short_code_01.txt",
        "short_prose_01.txt",
        "short_prose_02.txt",
        "long_code_01.txt",
        "long_prose_01.txt",
        "long_prose_02.txt",
    }
    actual_names = {f.name for f in txt_files}
    assert actual_names == expected_files

    for f in txt_files:
        content = f.read_text(encoding="utf-8")
        h1 = hashlib.sha256(content.encode("utf-8")).hexdigest()
        h2 = hashlib.sha256(f.read_bytes()).hexdigest()
        assert h1 == h2

        if f.name.startswith("short_"):
            assert len(content) < 1000, f"{f.name} expected under 1k chars, got {len(content)}"
        elif f.name.startswith("long_"):
            assert len(content) > 8000, f"{f.name} expected over 8k chars, got {len(content)}"

