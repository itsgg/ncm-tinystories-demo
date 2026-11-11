#!/usr/bin/env python3
"""Download the four TinyStories checkpoint files used by demo.py."""

import argparse
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

from demo import CHECKPOINT_HASHES, CHECKPOINT_REVISION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    if sys.version_info < (3, 11):
        parser.error("Python 3.11 or newer is required")
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    base = "https://huggingface.co/hazyresearch/ncm-checkpoints/resolve/"
    for name, expected in CHECKPOINT_HASHES.items():
        destination = args.directory / name
        if not destination.exists():
            temporary = destination.with_suffix(destination.suffix + ".part")
            url = f"{base}{CHECKPOINT_REVISION}/tinystories/{name}"
            print(f"Downloading {name}", flush=True)
            with urllib.request.urlopen(url, timeout=60) as response:
                with temporary.open("wb") as output:
                    shutil.copyfileobj(response, output)
            with temporary.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != expected:
                temporary.unlink()
                raise ValueError(f"Unexpected SHA-256 for {name}: {actual}")
            temporary.replace(destination)
        with destination.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Unexpected SHA-256 for {name}: {actual}")
        print(f"Verified {name}", flush=True)


if __name__ == "__main__":
    main()
