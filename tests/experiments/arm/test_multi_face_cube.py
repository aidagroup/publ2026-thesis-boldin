"""Unit tests for the multi-face cube config (pure dataclass validation)."""

import math

import pytest

from callosum.experiments.arm.skills_cube_config import (
    FACE_NORMALS,
    LAYER_JOINT_LIMITS_DEG,
    LayerSpec,
    MultiFaceCubeConfig,
    default_layers,
)


def layers_with(replacement: LayerSpec) -> tuple[LayerSpec, ...]:
    """The default six layers with `replacement` swapped in for its face."""
    return tuple(
        replacement if layer.face == replacement.face else layer for layer in default_layers()
    )


# --- defaults ---------------------------------------------------------------------------


def test_defaults_cover_six_distinct_faces_with_normal_axes() -> None:
    cfg = MultiFaceCubeConfig()
    faces = [layer.face for layer in cfg.layers]
    assert sorted(faces) == sorted(FACE_NORMALS)  # six distinct faces
    for layer in cfg.layers:
        assert layer.axis == FACE_NORMALS[layer.face]
        assert layer.detents_deg == (0.0, 90.0, 180.0)
        assert layer.lock_unless_held
        assert layer.friction == pytest.approx(20.0)


def test_default_geometry_matches_the_turntable_cube() -> None:
    cfg = MultiFaceCubeConfig()
    assert cfg.cube_half_size == pytest.approx(0.0285)
    assert cfg.layer_thickness == pytest.approx(2 * 0.0285 / 3)  # a real third of the edge
    assert 0 < cfg.layer_gap < cfg.layer_thickness


# --- layer validation -------------------------------------------------------------------


def test_layer_axis_must_be_a_unit_vector_collinear_with_the_face_normal() -> None:
    # Anti-parallel is fine (it only flips the joint's positive direction)...
    LayerSpec("U", (0.0, 0.0, -1.0))
    # ...but scaled, diagonal and wrong-face axes are not.
    with pytest.raises(ValueError, match="unit vector"):
        LayerSpec("U", (0.0, 0.0, 2.0))
    with pytest.raises(ValueError, match="collinear"):
        LayerSpec("U", (math.sqrt(0.5), math.sqrt(0.5), 0.0))  # unit, but not a face axis
    with pytest.raises(ValueError, match="collinear"):
        LayerSpec("U", (1.0, 0.0, 0.0))  # the R normal, not U's
    with pytest.raises(ValueError, match="face"):
        LayerSpec("X", (1.0, 0.0, 0.0))


def test_layer_rejects_bad_friction_and_detents() -> None:
    with pytest.raises(ValueError, match="friction"):
        LayerSpec("F", (0.0, 1.0, 0.0), friction=-1.0)
    with pytest.raises(ValueError, match="detents"):
        LayerSpec("F", (0.0, 1.0, 0.0), detents_deg=(0.0, 200.0))  # outside the limits
    with pytest.raises(ValueError, match="detents"):
        LayerSpec("F", (0.0, 1.0, 0.0), detents_deg=(90.0, 90.0))  # not strictly increasing
    with pytest.raises(ValueError, match="detents"):
        LayerSpec("F", (0.0, 1.0, 0.0), detents_deg=(-90.0, 90.0))


def test_detents_and_limits_convert_to_radians() -> None:
    layer = LayerSpec("R", (1.0, 0.0, 0.0))
    assert layer.detents_rad == (0.0, math.pi / 2, math.pi)
    assert MultiFaceCubeConfig().joint_limits_rad == (0.0, math.pi)
    assert LAYER_JOINT_LIMITS_DEG == (0.0, 180.0)


# --- cube config validation -------------------------------------------------------------


def test_config_requires_each_face_exactly_once() -> None:
    with pytest.raises(ValueError, match="exactly once"):
        # Duplicate U (and a missing face) must not pass.
        MultiFaceCubeConfig(
            layers=layers_with(LayerSpec("U", (0.0, 0.0, 1.0))) + (LayerSpec("U", (0.0, 0.0, 1.0)),)
        )
    with pytest.raises(ValueError, match="exactly once"):
        MultiFaceCubeConfig(layers=default_layers()[:5])  # a face missing


def test_config_rejects_unsane_geometry() -> None:
    with pytest.raises(ValueError, match="cube_half_size"):
        MultiFaceCubeConfig(cube_half_size=0.0)
    with pytest.raises(ValueError, match="layer_thickness"):
        MultiFaceCubeConfig(layer_thickness=2 * 0.0285)  # as thick as the whole cube
    with pytest.raises(ValueError, match="layer_thickness"):
        MultiFaceCubeConfig(layer_thickness=0.0)
    with pytest.raises(ValueError, match="layer_gap"):
        MultiFaceCubeConfig(layer_gap=-0.001)
    with pytest.raises(ValueError, match="layer_gap"):
        MultiFaceCubeConfig(layer_gap=0.02)  # gap as large as the layer is thick


def test_config_validates_layers_it_is_given() -> None:
    with pytest.raises(ValueError, match="collinear"):
        MultiFaceCubeConfig(layers=layers_with(LayerSpec("U", (1.0, 0.0, 0.0))))


def test_layer_lookup() -> None:
    cfg = MultiFaceCubeConfig()
    assert cfg.layer("D").axis == (0.0, 0.0, -1.0)
    with pytest.raises(KeyError):
        cfg.layer("X")


def test_custom_friction_and_detents_are_accepted() -> None:
    layer = LayerSpec("B", (0.0, -1.0, 0.0), friction=50.0, detents_deg=(0.0, 180.0))
    cfg = MultiFaceCubeConfig(layers=layers_with(layer))
    assert cfg.layer("B") == layer
    assert cfg.layer("B").friction == 50.0
    assert cfg.layer("B").detents_rad == (0.0, math.pi)
