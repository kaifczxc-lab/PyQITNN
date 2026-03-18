import sys
from pathlib import Path

from setuptools import setup


ROOT = Path(__file__).resolve().parent
CSRC = ROOT / "csrc"

_IS_WIN = sys.platform == "win32"

from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch
import torch.utils.cpp_extension as _cpp_ext

if torch.version.cuda is None:
    raise RuntimeError(
        "pyqitnn requires a CUDA-enabled PyTorch build in the build environment. "
        "Detected CPU-only torch. Build with the same CUDA PyTorch env you use at runtime."
    )

if not getattr(_cpp_ext, "CUDA_HOME", None):
    raise RuntimeError(
        "pyqitnn requires a local CUDA toolkit for compilation. "
        "Set CUDA_HOME/CUDA_PATH or add nvcc to PATH before building."
    )

#====================
# bypass CUDA version check
# PyTorch bundles its own CUDA runtime, so the system nvcc version
# does not need to match exactly. nvcc 13.x can compile code that
# runs with PyTorch's bundled CUDA 12.x runtime.
#====================
_cpp_ext._check_cuda_version = lambda *a, **kw: None

#====================
# host compiler flags
#====================

if _IS_WIN:
    cxx_flags = ["/O2", "/std:c++20"]
else:
    cxx_flags = ["-O2", "-std=c++20"]

#====================
# nvcc flags: optimization + gencode for all major GPU architectures
#====================

nvcc_flags = ["-O2"]

# detect supported GPU architectures from nvcc
_target_archs = [
    (75, "sm_75"),   # Turing (RTX 2000, T4)
    (80, "sm_80"),   # Ampere (A100)
    (86, "sm_86"),   # Ampere (RTX 3060-3090)
    (89, "sm_89"),   # Ada    (RTX 4000)
    (90, "sm_90"),   # Hopper (H100)
    (100, "sm_100"), # Blackwell
    (120, "sm_120"), # next-gen
]

# query nvcc for supported architectures
import subprocess as _sp
_supported = set()
try:
    _r = _sp.run(["nvcc", "--list-gpu-arch"], capture_output=True, text=True, timeout=10)
    for line in _r.stdout.strip().splitlines():
        tok = line.strip().replace("compute_", "")
        if tok.isdigit():
            _supported.add(int(tok))
except Exception:
    # fallback: assume a reasonable set
    _supported = {75, 80, 86, 89, 90}

_last_cc = None
for cc, sm in _target_archs:
    if cc in _supported:
        nvcc_flags.append(f"-gencode=arch=compute_{cc},code={sm}")
        _last_cc = cc

# PTX fallback for forward compatibility
if _last_cc is not None:
    nvcc_flags.append(f"-gencode=arch=compute_{_last_cc},code=compute_{_last_cc}")

#====================
# link against PyTorch's bundled CUDA libs, not the system CUDA toolkit
# this ensures _C.pyd loads cublas64_12.dll from torch/lib/
# even when the system has CUDA 13.x installed
#====================
_torch_lib = str(Path(torch.__file__).parent / "lib")

ext_modules = [
    CUDAExtension(
        name="pyqitnn._C",
        sources=[
            "src/pyqitnn_ext.cpp",
            "csrc/qitnn_cuda.cu",
        ],
        include_dirs=[
            str(CSRC / "include"),
        ],
        define_macros=[
            ("QITNN_BUILD", "1"),
        ],
        libraries=["cublas"],
        library_dirs=[_torch_lib],
        extra_compile_args={
            "cxx": cxx_flags,
            "nvcc": nvcc_flags,
        },
    )
]
cmdclass = {"build_ext": BuildExtension}


#====================
# setup (metadata lives in pyproject.toml, only build config here)
#====================

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
