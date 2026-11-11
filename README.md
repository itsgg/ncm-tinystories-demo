# TinyStories NCM demo

Generate three stories, inspect their twelve code levels, then fix the
first three levels of one sample and generate two variants of the rest.
The model and tokenizer are the released
[HazyResearch NCM](https://github.com/HazyResearch/ncm) checkpoints.

`residual.py` is the four-number example from the accompanying post. It
needs only Python and demonstrates residual quantization, not text
generation.

## Setup

The model demo requires Python 3.11 or newer. The captured environment uses
Python 3.12.15 on Linux and the CPU packages pinned in `requirements.txt`.

```sh
git clone https://github.com/itsgg/ncm-tinystories-demo.git
cd ncm-tinystories-demo
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

git clone https://github.com/HazyResearch/ncm.git ncm-source
git -C ncm-source checkout c70274273f489abc73832b2b98584a8483b6b203

python download.py checkpoints
python demo.py --source ncm-source --checkpoint-dir checkpoints --output my-run.json
```

`results.json` is the captured run; writing to `my-run.json` keeps it for
comparison.

The checkpoint download is about 578 MB. It uses a fixed Hugging Face
revision. The downloader and demo both check SHA-256 hashes; the demo also
checks that the upstream checkout is clean and at the expected commit.

The demo uses CPU, two threads and batch size one. Budget several GB of RAM
for the Python process. The captured run is limited to 3 GB by its process
manager. The script does not establish those operating-system limits.

## Reading the output

Each entry in `runs` has a seed, the full generated code sequence and twelve
decoded levels. Every decode receives a full-length cumulative latent
representation. Early text can be incoherent; it is not an outline supplied
to the next generation step. `raw_text` retains special tokens; `text`
removes padding and stops at EOS.

`fixed_prefix_variants` holds the first 25 codes, corresponding to the first
three levels, of the seed-zero sample. The remaining nine levels are
sampled with seeds one and two. Assertions check the fixed codes and confirm
that later codes change.

This adapts the experiment in
[the paper's Appendix H](https://arxiv.org/html/2610.14162v1#A8). The paper
fixes coarse codes encoded from supplied stories. This example fixes codes
from a generated story.

The JSON includes environment versions and artifact hashes. Exact stories
can differ between processors and PyTorch builds. Timings have no warmup,
exclude intermediate diagnostic decodes and are not benchmark results.

The numerical example runs separately:

```sh
python residual.py
```
