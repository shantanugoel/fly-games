"""The per-title contract for the fly lab.

Same shape as the Jev lab, but the policy is a **fly brain** instead of an LLM:

  game.observe(env, info, memory, hold, prev_action) -> observation (structured facts)
  game.fly_encoder(client)                -> SensoryEncoder
  fly_encoder.encode(observation)         -> inject, input_labels
  client.decide(inject, input_labels)     -> BrainDecision (fly's descending output)
  readout.decode(feature)                 -> coarse fly decision + probabilities
  game.fly_expand(coarse, observation)    -> fine controller action key
  game.advance(env, fine_action, hold, obs) -> AdvanceResult

The brain is never trained; only the per-game readout changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from fly_games.types import Action, AdvanceResult, GameMeta

if TYPE_CHECKING:
    from fly_games.brain.encoder import SensoryEncoder


class Game(ABC):
    """One playable title. Registered in `fly_games.catalog`."""

    meta: GameMeta
    actions: tuple[Action, ...]

    # --- the shared engine contract (emulator) ---
    @abstractmethod
    def create_env(self):
        """Return a live emulator/session for this game."""

    @abstractmethod
    def reset(self, env, seed: int):
        """Reset and return (frame, info)."""

    @abstractmethod
    def observe(self, env, info, memory, hold_frames: int, previous_action: str):
        """Structured observation consumed by the fly encoder."""

    def scene(self, observation: dict) -> dict:
        """UI facts/overlays/text. Default: fallback to the observation text."""
        return {"facts": [], "overlays": [], "text": str(observation.get("text", ""))}

    def stats(self, observation: dict, info: dict, session: dict):
        return []

    @abstractmethod
    def advance(self, env, action_key: str, frames: int, observation: dict) -> AdvanceResult:
        """Hold `action_key` up to `frames`, optionally interrupting early."""

    @abstractmethod
    def scripted(self, observation: dict) -> tuple[str, dict]:
        """Deterministic baseline: the ground-truth controller used for the
        lab's scripted mode AND as the label source for training the fly readout."""

    # --- fly-specific contract (the brain policy) ---
    @abstractmethod
    def fly_encoder(self, client) -> SensoryEncoder:
        """Return the sensory encoder for this game (drives LPLC2/LC4/LPLC1/LC10a)."""

    @abstractmethod
    def fly_coarse_actions(self) -> tuple[str, ...]:
        """Coarse decisions the fly's command output can reliably express."""

    @abstractmethod
    def fly_coarse_of(self, fine_action: str, observation: dict) -> str:
        """Map a fine controller action (from the scripted policy) to the coarse
        fly decision it corresponds to. Used as the training label."""

    @abstractmethod
    def fly_expand(self, coarse: str, observation: dict, fine_hint: str | None = None) -> str:
        """Expand a coarse fly decision into a fine controller action key (buttons),
        using game knowledge and the current observation."""

    @abstractmethod
    def fly_hand_decode(self, command: dict, observation: dict) -> tuple[str, dict]:
        """Zero-shot decode of the descending command rates into a coarse decision.

        Used until a readout has been trained (`fly-games train`). `command` is
        {group: {"count":, "rate":}}. Returns (coarse_action, probabilities).
        """

    @property
    def level_label(self) -> str | None:
        """Human label for the current level/floor, for the episode archive."""
        return None

    # --- shared support ---
    def new_memory(self):
        """Episode-local memory (transition history, landing tracking, …)."""
        return

    def finish_memory(self, memory, before, env, info, action, result) -> None:
        pass

    def close(self, env) -> None:
        close = getattr(env, "close", None)
        if close:
            close()

    def completed(self, info: dict) -> bool:
        return False

    def catalog_entry(self) -> dict:
        meta = self.meta
        return {
            "id": meta.id,
            "title": meta.title,
            "short_name": meta.short_name,
            "platform_id": meta.platform_id,
            "platform_label": meta.platform_label,
            "cartridge": meta.cartridge,
        }

    def button_keys(self) -> tuple[str, ...]:
        return self.meta.button_keys

    def _action_index(self, action_key: str) -> int:
        for i, action in enumerate(self.actions):
            if action.key == action_key:
                return i
        raise KeyError(action_key)
