"""fly-games: play mario, kung-fu/spartan-x and doom with a real fruit-fly brain.

The MaleCNS v1.0 connectome (166,700 neurons, 25.6M synapses) is a *frozen* spiking
network. It is never trained. Each game contributes only two small pieces that sit on
top of the connectome:

    input  ->  an **encoder** that turns structured game facts into voltage on the fly's
               own sensory / visual-projection neuron types (looming -> LPLC2,
               threat -> LC4, small approaching -> LPLC1, chase -> LC10a).
    output ->  a **readout** that decodes the fly's descending command neurons (escape,
               steer, forward, backward, punch, kick) into a game controller action.

The fly.ai reservoir philosophy: the brain is frozen, only the encoder + readout change
per game.
"""

__version__ = "0.1.0"

from .types import Action, AdvanceResult, GameMeta, Stat

__all__ = ["Action", "AdvanceResult", "GameMeta", "Stat", "__version__"]
