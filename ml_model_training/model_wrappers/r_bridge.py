"""R-subprocess utilities for mgcv-backed GAM wrappers.

VMD column-name parsing lives in utils/feature_naming.py (parse_vmd_column);
`san` is re-exported here for backward compatibility, since R-backed
wrappers import it from this module.
"""

import functools
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from utils.feature_naming import san


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


# Prepended to every generated script so R packages installed into the user
# library (not the system library under Program Files, which is typically
# read-only) are found regardless of which account/session runs Rscript.
LIBPATH_HEADER = '.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))\n'


def run_r(script: str, timeout: int = 1200) -> str:
    """Execute an R script string via Rscript; return combined stdout + stderr."""
    rscript = find_rscript()
    tmp = tempfile.NamedTemporaryFile(suffix=".R", delete=False, mode="w", encoding="utf-8")
    tmp.write(LIBPATH_HEADER)
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
