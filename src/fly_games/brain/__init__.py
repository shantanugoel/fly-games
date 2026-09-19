"""The fly connectome client, sensory encoders, and readouts."""

from .client import (
    CHANNEL_MEANING,
    CHANNELS,
    COMMAND_GROUPS,
    DEFAULT_PROBE_SIZE,
    NO_PROBE,
    POINT_BYTES,
    SUPERCLASS_COLORS,
    BrainDecision,
    FlyBrainClient,
    projection_axes,
    shared_client,
)
from .encoder import ENCODER_PARAMS, NullEncoder, SensoryEncoder, lateral_sides
from .readout import FlyReadout, ReadoutInfo

__all__ = [
    "CHANNELS",
    "CHANNEL_MEANING",
    "COMMAND_GROUPS",
    "DEFAULT_PROBE_SIZE",
    "ENCODER_PARAMS",
    "NO_PROBE",
    "POINT_BYTES",
    "SUPERCLASS_COLORS",
    "BrainDecision",
    "FlyBrainClient",
    "FlyReadout",
    "NullEncoder",
    "ReadoutInfo",
    "SensoryEncoder",
    "lateral_sides",
    "projection_axes",
    "shared_client",
]
