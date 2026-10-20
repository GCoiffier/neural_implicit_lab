from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("implicitlab")
except PackageNotFoundError:  # package not installed (e.g. running from a raw source checkout)
    __version__ = "unknown"

from .data.geometry import load_geometry
from .data.sampler import PointSampler
from .data import fields
from .data import sampling_strategies

from .nn.implicit_model import load_model
from . import nn
from . import training
from . import utils
from . import diff_op
from .queries import queries, visualize
