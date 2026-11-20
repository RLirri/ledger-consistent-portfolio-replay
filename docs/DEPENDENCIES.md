# Dependencies

Observed primary interpreter: Python 3.11.11. Direct dependencies: NumPy 2.4.1 (BSD-3-Clause), pandas 2.3.3 (BSD-3-Clause), PyArrow 23.0.0 (Apache-2.0), Matplotlib 3.10.8 (Matplotlib/PSF-compatible license), Pillow 12.1.0 (MIT-CMU). These are installed dependencies, not vendored code; their own notices apply.

requirements.txt pins direct scientific/figure versions. pyproject.toml declares core and figure extras. Setuptools (MIT) is a bounded build dependency, not a complete hashed binary lock. Transitive builds vary by platform. V1 optional acquisition/provider packages are outside the validated V2 core environment.

Standard-library unittest is used. Batch resource controls require POSIX. No paid service or telemetry is required. Font licenses remain the user's responsibility; no font files are distributed.

Portable installation was checked offline with setuptools65.5.0 and wheel0.46.3; both build tools are pinned in requirements.txt. License metadata uses the LICENSE file form compatible with that tested backend.
