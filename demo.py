#!/usr/bin/env python3
"""Inspect the released TinyStories NCM model on CPU, batch size one."""

import argparse
import hashlib
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

SOURCE_REVISION = "c70274273f489abc73832b2b98584a8483b6b203"
CHECKPOINT_REVISION = "d342ce8aaea58830eef3b8ad473664673e41f692"
CHECKPOINT_HASHES = {
    "generator.pt": "0acaf4f795390f9f7ccc8f84047867619a1d00e291681e299b855d7e563be6ff",
    "tokenizer.pt": "76e26003719db32dc3d542518951c286be07d6a5881f29ad5152d299764b0f81",
    "bpe_vocab_8192.pickle": "fb176b4c68108e4b5ef61b9ff2af3731600124e3e160970a6bd78cd0a5d3ff6c",
    "bpe_merges_8192.pickle": "0d4cab13f2e7b684b75eaa290a52e596ba87744ef13d943cf3a50a5a8e080ec1",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    if sys.version_info < (3, 11):
        parser.error("Python 3.11 or newer is required")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    checkpoint_dir = args.checkpoint_dir.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != SOURCE_REVISION:
        parser.error(f"Expected upstream revision {SOURCE_REVISION}, got {revision}")
    if subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"], text=True
    ).strip():
        parser.error("The upstream source checkout must be clean")
    # Validate every artifact before loading checkpoint or BPE pickle data.
    fingerprints = {}
    for name, expected in CHECKPOINT_HASHES.items():
        with (checkpoint_dir / name).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            parser.error(f"Unexpected SHA-256 for {name}: {actual}")
        fingerprints[name] = actual
    sys.path.insert(0, str(source))

    import torch
    from omegaconf import OmegaConf
    from models.next_concept import NextConceptTransformer
    from tokenization.vqvae_tokenizer import VQVAETokenizer

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cfg = OmegaConf.load(source / "config/eval/tinystories.yaml")
    cfg.tokenizer.vqvae.checkpoint_path = str(checkpoint_dir / "tokenizer.pt")
    cfg.tokenizer.pre_tokenizer.vocab_file_path = str(
        checkpoint_dir / "bpe_vocab_8192.pickle"
    )
    cfg.tokenizer.pre_tokenizer.merges_file_path = str(
        checkpoint_dir / "bpe_merges_8192.pickle"
    )
    print("Loading TinyStories tokenizer and generator on CPU", flush=True)
    tokenizer = VQVAETokenizer.from_config(cfg.tokenizer)
    tokenizer.vqvae.eval()
    model = NextConceptTransformer(vqvae_local=tokenizer.vqvae, config=cfg.model)
    checkpoint = torch.load(
        checkpoint_dir / "generator.pt", map_location="cpu", weights_only=False
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    initial_distribution = checkpoint["inference"]["level0_unigram"].float()
    del checkpoint
    levels = list(tokenizer.vqvae.vq.levels)
    assert levels == [1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128]
    prefix = torch.full((1, 1), tokenizer.pre_tokenizer.eos_token_id)
    sample_level = model._sample_level

    def generate(seed, fixed_codes=None, keep_levels=0):
        torch.manual_seed(seed)
        first = (
            torch.multinomial(initial_distribution, levels[0], replacement=True)
            .reshape(1, levels[0])
            if fixed_codes is None else fixed_codes[:, :levels[0]].clone()
        )
        generated_level = 0

        def sample(logits, temperature, top_k, top_p):
            nonlocal generated_level
            generated_level += 1
            if fixed_codes is not None and generated_level < keep_levels:
                start = sum(levels[:generated_level])
                return fixed_codes[:, start:start + levels[generated_level]].clone()
            return sample_level(logits, temperature, top_k, top_p)

        model._sample_level = sample
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                codes = model.generate(
                    first, prefix, temperature=1.0, top_k=0, top_p=1.0,
                    cfg_scale=0.0, use_kv_cache=False,
                )
        finally:
            model._sample_level = sample_level
        seconds = time.perf_counter() - started
        assert generated_level == len(levels) - 1
        assert tuple(codes.shape) == (1, sum(levels))
        if fixed_codes is not None:
            n_fixed = sum(levels[:keep_levels])
            assert torch.equal(codes[:, :n_fixed], fixed_codes[:, :n_fixed])
            assert not torch.equal(codes[:, n_fixed:], fixed_codes[:, n_fixed:])

        # Decode full-length cumulative reconstructions, as in the paper.
        # Avoid idx_to_zhats's aliased first return and shortened z_levels.
        with torch.inference_mode():
            vq = tokenizer.vqvae.vq
            accumulated = torch.zeros(1, levels[-1], vq.embed_dim)
            offset = 0
            trace = []
            for level, length in enumerate(levels):
                contribution = vq.embeddings[level](codes[:, offset:offset + length])
                if level != len(levels) - 1:
                    contribution = vq._up_to_full(contribution, level, levels[-1])
                contribution = vq._apply_quant_resi(contribution, level)
                accumulated = accumulated + contribution
                snapshot = accumulated.clone()
                token_ids = tokenizer.vqvae.decoder(snapshot).argmax(-1)[0].tolist()
                assert len(token_ids) == 128
                raw_text = tokenizer.pre_tokenizer.decode(token_ids)
                eos = tokenizer.pre_tokenizer.eos_token_id
                visible_ids = token_ids[:token_ids.index(eos)] if eos in token_ids else token_ids
                text = tokenizer.pre_tokenizer.decode(visible_ids, skip_special_tokens=True)
                trace.append({
                    "level": level + 1, "positions": length,
                    "decoded_positions": len(token_ids), "text": text,
                    "raw_text": raw_text, "token_ids": token_ids,
                })
                offset += length
        assert len(trace) == 12 and len(trace[-1]["token_ids"]) == 128
        with torch.inference_mode():
            reference_ids = tokenizer.vqvae.decode(codes).argmax(-1)[0].tolist()
        assert trace[-1]["token_ids"] == reference_ids
        record = {
            "seed": seed, "fixed_levels": keep_levels,
            "generator_calls": generated_level,
            "generation_seconds_excluding_decodes": seconds,
            "codes": codes[0].tolist(), "levels": trace,
        }
        print(f"seed={seed}, fixed_levels={keep_levels}, calls={generated_level}", flush=True)
        print(trace[-1]["text"], flush=True)
        return codes, record

    # Predeclared seeds: retain every output, including unfinished stories.
    base_codes, base = generate(0)
    runs = [base, generate(1)[1], generate(42)[1]]
    variants = [generate(seed, base_codes, keep_levels=3)[1] for seed in (1, 2)]
    result = {
        "source_revision": revision, "checkpoint_sha256": fingerprints,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "python": platform.python_version(), "torch": torch.__version__,
        "device": "cpu", "batch_size": 1, "threads": 2,
        "timing_note": "No warmup; diagnostic CPU timings are not a benchmark.",
        "decode_note": "Full-length cumulative latents; fresh snapshot per level.",
        "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
            1 if platform.system() == "Darwin" else 1024
        ),
        "level_lengths": levels, "runs": runs, "fixed_prefix_variants": variants,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
