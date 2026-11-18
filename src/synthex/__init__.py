"""SynthEx strategy-first synthesis planning."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("synthex")
except PackageNotFoundError:  # source checkout without installation
    __version__ = "0.1.0"

__all__ = ["__version__"]
