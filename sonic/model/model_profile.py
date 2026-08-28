"""Released SONIC model families and their exact flat tensor contracts.

The encoder, decoder, and observation profile are an atomic set.  This module
keeps that choice separate from the physical G1 embodiment profile in
``robot_profile.py``: every released decoder still predicts 29 IsaacLab joints.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SonicModelProfile:
    name: str
    encoder_input_dim: int
    token_dim: int = 64
    decoder_input_dim: int = 994
    decoder_output_dim: int = 29
    g1_future_frames: int = 10
    g1_future_stride: int = 5
    smpl_future_frames: int = 10
    smpl_future_stride: int = 1
    heading_normalized: bool = False
    include_root_height_terms: bool = True
    encoder_anchor_order: str = "single_then_multi"

    def __post_init__(self):
        if self.encoder_anchor_order not in {
            "single_then_multi",
            "multi_then_single",
        }:
            raise ValueError(
                "encoder_anchor_order must be single_then_multi or "
                "multi_then_single"
            )

    @property
    def expected_shapes(self):
        return (
            (1, self.encoder_input_dim),
            (1, self.token_dim),
            (1, self.decoder_input_dim),
            (1, self.decoder_output_dim),
        )


MODEL_PROFILES = {
    "default": SonicModelProfile(
        "default", 1762, encoder_anchor_order="single_then_multi"
    ),
    "low_latency": SonicModelProfile(
        "low_latency",
        1247,
        g1_future_stride=1,
        smpl_future_frames=4,
        heading_normalized=False,
        include_root_height_terms=False,
        encoder_anchor_order="single_then_multi",
    ),
    "sonic_v1_1": SonicModelProfile(
        "sonic_v1_1",
        1751,
        heading_normalized=True,
        include_root_height_terms=False,
        encoder_anchor_order="multi_then_single",
    ),
}


def get_model_profile(name: str) -> SonicModelProfile:
    key = str(name).strip().lower().replace("-", "_").replace(".", "_")
    aliases = {
        "release": "default",
        "sonic_release": "default",
        "lowlatency": "low_latency",
        "v1_1": "sonic_v1_1",
        "sonic_1_1": "sonic_v1_1",
    }
    key = aliases.get(key, key)
    try:
        return MODEL_PROFILES[key]
    except KeyError as exc:
        raise ValueError(
            f"unknown SONIC model profile {name!r}; choose "
            f"{sorted(MODEL_PROFILES)}"
        ) from exc
