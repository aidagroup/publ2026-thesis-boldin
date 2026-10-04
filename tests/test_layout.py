"""Pure-math checks of the arm layout (no mani_skill needed)."""

import math

import pytest

from callosum.configs.layout import ArmLayout, base_xy_yaw


@pytest.mark.parametrize("azimuth", [-math.pi, -math.pi / 2, 0.0, 0.7, math.pi / 2])
def test_base_faces_table_centre(azimuth):
    x, y, yaw = base_xy_yaw(0.3, azimuth)
    # Robot "forward" is -y in its base frame; rotated by yaw it is (sin(yaw), -cos(yaw)).
    forward = (math.sin(yaw), -math.cos(yaw))
    to_centre = (-x / 0.3, -y / 0.3)
    assert forward == pytest.approx(to_centre, abs=1e-9)


def test_default_layout_arms_are_90_degrees_apart():
    layout = ArmLayout()
    assert abs(layout.azimuth_b - layout.azimuth_a) == pytest.approx(math.pi / 2)
    assert base_xy_yaw(layout.radius_a, layout.azimuth_a)[:2] != pytest.approx(
        base_xy_yaw(layout.radius_b, layout.azimuth_b)[:2]
    )
