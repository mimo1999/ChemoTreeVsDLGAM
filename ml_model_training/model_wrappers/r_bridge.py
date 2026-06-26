"""Shared R-subprocess utilities for mgcv-backed GAM wrappers."""

import functools
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

# VMD feature-name patterns shared across all R-backed wrappers.
RE_X = re.compile(r"^[Xx]_(.+)_(\d+)$")   # value    X_<itemid>_<bin>
RE_M = re.compile(r"^[Mm]_(.+)_(\d+)$")   # missing  M_<itemid>_<bin>
RE_D = re.compile(r"^[Dd]_(.+)_(\d+)$")   # delta    D_<itemid>_<bin>


def san(name: str) -> str:
    """Sanitise a feature name into a safe R identifier fragment."""
    return re.sub(r"[^A-Za-z0-9]", "_", str(name))


@functools.lru_cache(maxsize=1)
def find_rscript() -> str:
    """Locate the Rscript executable at runtime.

    Resolution order:
    1. ``RSCRIPT`` env var — full path override, checked first.
    2. ``shutil.which("Rscript")`` — works when R is on the system PATH.
    3. ``R_HOME`` env var → bin/x64/Rscript.exe (Windows) or bin/Rscript.
    4. Scan ``Program Files\\R\\`` on Windows; picks the lexicographically
       newest version directory.

    Raises ``RuntimeError`` with an actionable message if R cannot be found.
    """
    explicit = os.environ.get("RSCRIPT")
    if explicit:
        return explicit

    on_path = shutil.which("Rscript")
    if on_path:
        return on_path

    r_home = os.environ.get("R_HOME")
    if r_home:
        exe = "Rscript.exe" if os.name == "nt" else "Rscript"
        for sub in (Path(r_home) / "bin" / "x64", Path(r_home) / "bin"):
            p = sub / exe
            if p.exists():
                return str(p)

    if os.name == "nt":
        for prog_env in ("PROGRAMFILES", "PROGRAMFILES(X86)"):
            prog = os.environ.get(prog_env, "")
            r_root = Path(prog) / "R"
            if r_root.is_dir():
                for v in sorted(r_root.iterdir(), reverse=True):
                    for sub in (v / "bin" / "x64", v / "bin"):
                        p = sub / "Rscript.exe"
                        if p.exists():
                            return str(p)

    raise RuntimeError(
        "Rscript not found. Install R and ensure it is on PATH, or set the "
        "RSCRIPT environment variable to the full path of Rscript[.exe]."
    )


def run_r(script: str, timeout: int = 1200) -> str:
    """Execute an R script string via Rscript; return combined stdout + stderr."""
    rscript = find_rscript()
    tmp = tempfile.NamedTemporaryFile(suffix=".R", delete=False, mode="w", encoding="utf-8")
    tmp.write(script)
    tmp.close()
    proc = subprocess.run(
        [rscript, "--vanilla", tmp.name],
        capture_output=True, text=True, timeout=timeout,
    )
    try:
        os.unlink(tmp.name)
    except OSError:
        pass
    return proc.stdout + ("\n" + proc.stderr if proc.stderr.strip() else "")
