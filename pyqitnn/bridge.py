from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path


_NATIVE = None
_DLL_HANDLES: list[object] = []

_IS_WIN = sys.platform == "win32"
_IS_LINUX = sys.platform.startswith("linux")

_EXT_GLOB = "_C*.pyd" if _IS_WIN else "_C*.so"


#====================
# path resolution
#====================

def _torch_lib_dir() -> Path | None:
    try:
        import torch
    except Exception:
        return None
    return Path(torch.__file__).resolve().parent / "lib"


def _cuda_bin_dir() -> Path | None:
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        base = Path(cuda_path)
        for rel in (Path("bin") / "x64", Path("bin")):
            p = base / rel
            if p.exists():
                return p

    if _IS_WIN:
        base = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
        if base.exists():
            candidates = sorted(base.glob("v*"), reverse=True)
            for c in candidates:
                for rel in (Path("bin") / "x64", Path("bin")):
                    p = c / rel
                    if p.exists():
                        return p

    if _IS_LINUX:
        for d in ("/usr/local/cuda/bin", "/opt/cuda/bin"):
            p = Path(d)
            if p.exists():
                return p

    return None


def _candidate_native_paths() -> list[Path]:
    pkg = Path(__file__).resolve().parent
    candidates = sorted(pkg.glob(_EXT_GLOB))

    build_dir = pkg.parent / "build"
    if build_dir.exists():
        for candidate in sorted(build_dir.glob("lib*"), reverse=True):
            candidates.extend(sorted((candidate / "pyqitnn").glob(_EXT_GLOB)))

    seen: set[Path] = set()
    out: list[Path] = []
    for path in candidates:
        if path.exists() and path not in seen:
            seen.add(path)
            out.append(path)
    return out


def _pick_native_path() -> Path | None:
    candidates = _candidate_native_paths()
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime_ns)


#====================
# library search path setup
#====================

def _add_dll_dir(path: Path) -> None:
    if not path.exists():
        return
    if _IS_WIN and hasattr(os, "add_dll_directory"):
        _DLL_HANDLES.append(os.add_dll_directory(str(path)))
    else:
        os.environ["PATH"] = str(path) + os.pathsep + os.environ.get("PATH", "")
        if _IS_LINUX:
            os.environ["LD_LIBRARY_PATH"] = (
                str(path) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
            )


#====================
# runtime init and native module loading
#====================

def prepare_runtime() -> dict[str, object]:
    """Add CUDA and Torch library dirs to the search path.

    Called automatically on `import pyqitnn`.  The CUDA kernels are
    compiled directly into the _C extension, so there is no separate
    QITNN.dll / libQITNN.so to locate.
    """
    torch_lib = _torch_lib_dir()
    cuda_bin = _cuda_bin_dir()

    if torch_lib is not None:
        _add_dll_dir(torch_lib)
    if cuda_bin is not None:
        _add_dll_dir(cuda_bin)

    return {
        "torch_lib_dir": str(torch_lib) if torch_lib is not None else None,
        "cuda_bin_dir": str(cuda_bin) if cuda_bin is not None else None,
    }


def load_native():
    global _NATIVE
    if _NATIVE is not None:
        return _NATIVE

    import torch  # noqa: F401

    prepare_runtime()
    native_path = _pick_native_path()
    if native_path is None:
        _NATIVE = importlib.import_module("pyqitnn._C")
        return _NATIVE

    spec = importlib.util.spec_from_file_location("pyqitnn._C", native_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create import spec for native extension: {native_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["pyqitnn._C"] = module
    spec.loader.exec_module(module)
    _NATIVE = module
    return _NATIVE


def bridge_status() -> dict[str, object]:
    """Check native extension status. Attempts actual import to verify."""
    native_path = _pick_native_path()
    loadable = False
    load_error: str | None = None
    if native_path is not None:
        try:
            load_native()
            loadable = True
        except Exception as e:
            load_error = str(e)
    return {
        "stage": "python-wrapper",
        "native_found": native_path is not None,
        "native_loadable": loadable,
        "native_module_path": str(native_path) if native_path is not None else None,
        "load_error": load_error,
    }
