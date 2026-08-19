#!/usr/bin/env python
"""Create a source-only ML-DFI handoff ZIP from reviewed repository files."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from handoff_preflight import (  # noqa: E402
    git_handoff_candidates,
    package_exclusion_reason,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "dist" / "ml_dfi_handoff.zip",
        help="Destination ZIP path (default: dist/ml_dfi_handoff.zip).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing destination ZIP.",
    )
    args = parser.parse_args()

    preflight = subprocess.run(
        [sys.executable, str(ROOT / "handoff_preflight.py")],
        cwd=ROOT,
        check=False,
    )
    if preflight.returncode:
        print("Packaging aborted because handoff preflight failed.", file=sys.stderr)
        return preflight.returncode

    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        print(
            f"Destination already exists: {output}\n"
            "Use --overwrite or choose another --output path.",
            file=sys.stderr,
        )
        return 2
    output.parent.mkdir(parents=True, exist_ok=True)

    included: list[Path] = []
    excluded: list[tuple[Path, str]] = []
    for relative in git_handoff_candidates():
        source = ROOT / relative
        if not source.is_file():
            continue
        reason = package_exclusion_reason(relative)
        if reason:
            excluded.append((relative, reason))
        else:
            included.append(relative)

    manifest_lines = [
        "# ML-DFI handoff package manifest",
        "# SHA256  bytes  relative_path",
    ]
    for relative in sorted(included, key=lambda item: item.as_posix().lower()):
        source = ROOT / relative
        manifest_lines.append(
            f"{sha256_file(source)}  {source.stat().st_size}  {relative.as_posix()}"
        )
    manifest = "\n".join(manifest_lines) + "\n"

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output.stem}.",
            suffix=".tmp",
            dir=output.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)

        with zipfile.ZipFile(
            temporary_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for relative in included:
                archive.write(ROOT / relative, relative.as_posix())
            archive.writestr("PACKAGE_MANIFEST.txt", manifest)

        temporary_path.replace(output)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    print(f"Created: {output}")
    print(f"Included files: {len(included)}")
    print(f"Policy-excluded files: {len(excluded)}")
    print(f"Archive SHA256: {sha256_file(output)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
