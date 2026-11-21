"""Review candidate Git contents and validate checkpoint pointers before publication."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "evidence/package_manifest.json"
KERNEL = "outdoor_registration/vendor/geotransformer/modules/kpconv/dispositions/k_015_center_3D.ply"
COOKED_SUFFIXES = {".so", ".o", ".pyc", ".log", ".pkl", ".npy", ".bin", ".zip", ".gz", ".tar"}
SECRET = re.compile(rb"-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----|\b(?:gh[pousr]_[a-zA-Z0-9]{25,}|AKIA[A-Z0-9]{16})\b")
LOCAL_PATH = re.compile(rb"(?<![a-zA-Z0-9])/(?:home|tmp|mnt)/[a-zA-Z0-9_.-]+/")


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def tracked_names(*flags):
    return {p.decode() for p in git("ls-files", "-z", *flags).split(b"\0") if p}


def digest(contents):
    return hashlib.sha256(contents).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    options = parser.add_mutually_exclusive_group()
    options.add_argument("--write-manifest", action="store_true")
    options.add_argument("--staged", action="store_true")
    args = parser.parse_args()
    files = tracked_names("--cached")
    if not args.staged:
        files |= tracked_names("--others", "--exclude-standard")
    results = {}
    errors = []
    cfg = json.loads((ROOT / "configs/evaluation.json").read_text())
    checkpoints = {"weights/" + name: expected for name, expected in cfg["weights"].items()}
    for name in sorted(files - {MANIFEST}):
        path = ROOT / name
        if (path.is_symlink() or path.suffix in COOKED_SUFFIXES
                or ("build" in path.relative_to(ROOT).parts) or name.startswith("outputs/")
                or name == "configs/paths.local.json"
                or (name.startswith("data/") and name != "data/README.md")
                or path.suffix == ".npz" or (path.suffix == ".ply" and name != KERNEL)):
            errors.append(f"Unexpected local or generated asset: {name}")
            continue
        contents = path.read_bytes()
        sha = digest(contents)
        if name in checkpoints:
            if sha != checkpoints[name]:
                errors.append(f"Checkpoint SHA-256 mismatch: {name}")
            attr = git("check-attr", "filter", "--", name).decode().strip()
            if not attr.endswith(": lfs"):
                errors.append(f"Missing Git LFS filter: {name}")
            if args.staged:
                pointer = git("show", ":" + name)
                if pointer.splitlines() != [
                    b"version https://git-lfs.github.com/spec/v1",
                    b"oid sha256:" + sha.encode(),
                    b"size " + str(len(contents)).encode(),
                ]:
                    errors.append(f"Staged blob is not the expected LFS pointer: {name}")
        elif path.suffix != ".ply":
            if SECRET.search(contents):
                errors.append(f"Review possible credential in {name}")
            if LOCAL_PATH.search(contents):
                errors.append(f"Review absolute machine path in {name}")
            if args.staged and git("show", ":" + name) != contents:
                errors.append(f"Working file differs from staged content: {name}")
        results[name] = {"bytes": len(contents), "sha256": sha}
    if missing := set(checkpoints) - set(results):
        errors.append(f"Missing required checkpoints: {', '.join(sorted(missing))}")
    current = json.dumps(results, indent=2) + "\n"
    if args.staged and (MANIFEST not in files or git("show", ":" + MANIFEST).decode() != current):
        errors.append("Staged package manifest does not match staged files")
    if errors:
        raise SystemExit("\n".join(errors))
    if args.write_manifest:
        (ROOT / MANIFEST).write_text(current)
    print(f"Checked {len(results)} files and {len(checkpoints)} checkpoints" +
          (" with staged LFS pointers." if args.staged else "."))


if __name__ == "__main__":
    main()
