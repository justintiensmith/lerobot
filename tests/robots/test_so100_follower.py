#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lerobot.robots.so_follower import (
    SO100Follower,
    SO100FollowerConfig,
)
from lerobot.utils.constants import OBS_DEPTHS, OBS_IMAGES, OBS_MOTOR_CURRENTS, OBS_MOTOR_VELOCITIES, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features


def _make_bus_mock() -> MagicMock:
    """Return a bus mock with just the attributes used by the robot."""
    bus = MagicMock(name="FeetechBusMock")
    bus.is_connected = False

    def _connect():
        bus.is_connected = True

    def _disconnect(_disable=True):
        bus.is_connected = False

    bus.connect.side_effect = _connect
    bus.disconnect.side_effect = _disconnect

    @contextmanager
    def _dummy_cm():
        yield

    bus.torque_disabled.side_effect = _dummy_cm

    return bus


@pytest.fixture
def follower():
    bus_mock = _make_bus_mock()

    def _bus_side_effect(*_args, **kwargs):
        bus_mock.motors = kwargs["motors"]
        motors_order: list[str] = list(bus_mock.motors)

        def _sync_read(data_name, *_args, **_kwargs):
            if data_name == "Present_Position":
                return {motor: idx for idx, motor in enumerate(motors_order, 1)}
            if data_name == "Present_Current":
                return {motor: idx * 100 for idx, motor in enumerate(motors_order, 1)}
            if data_name == "Present_Velocity":
                return {motor: -idx * 10 for idx, motor in enumerate(motors_order, 1)}
            raise AssertionError(f"Unexpected sync_read data_name: {data_name}")

        bus_mock.sync_read.side_effect = _sync_read
        bus_mock.sync_write.return_value = None
        bus_mock.write.return_value = None
        bus_mock.disable_torque.return_value = None
        bus_mock.enable_torque.return_value = None
        bus_mock.is_calibrated = True
        return bus_mock

    with (
        patch(
            "lerobot.robots.so_follower.so_follower.FeetechMotorsBus",
            side_effect=_bus_side_effect,
        ),
        patch.object(SO100Follower, "configure", lambda self: None),
    ):
        cfg = SO100FollowerConfig(port="/dev/null")
        robot = SO100Follower(cfg)
        yield robot
        if robot.is_connected:
            robot.disconnect()


def test_connect_disconnect(follower):
    assert not follower.is_connected

    follower.connect()
    assert follower.is_connected

    follower.disconnect()
    assert not follower.is_connected


def test_get_observation(follower):
    follower.connect()
    obs = follower.get_observation()

    expected_keys = {f"{m}.pos" for m in follower.bus.motors}
    assert set(obs.keys()) == expected_keys

    for idx, motor in enumerate(follower.bus.motors, 1):
        assert obs[f"{motor}.pos"] == idx


def test_get_observation_includes_raw_motor_current_when_enabled(follower):
    follower.config.observe_motor_current = True

    follower.connect()
    obs = follower.get_observation()
    dataset_features = hw_to_dataset_features(follower.observation_features, OBS_STR, use_video=True)
    frame = build_dataset_frame(dataset_features, obs, prefix=OBS_STR)

    expected_position_keys = [f"{motor}.pos" for motor in follower.bus.motors]
    expected_current_keys = [f"{motor}.current" for motor in follower.bus.motors]
    expected_state = np.array([*range(1, len(follower.bus.motors) + 1)], dtype=np.float32)
    expected_currents = np.array(
        [idx * 100 for idx in range(1, len(follower.bus.motors) + 1)],
        dtype=np.int32,
    )

    assert set(obs) == {*expected_position_keys, "motor_currents"}
    assert follower.action_features == dict.fromkeys(expected_position_keys, float)
    assert dataset_features[f"{OBS_STR}.state"]["names"] == expected_position_keys
    assert dataset_features[OBS_MOTOR_CURRENTS] == {
        "dtype": "int32",
        "shape": (len(follower.bus.motors),),
        "names": expected_current_keys,
    }
    np.testing.assert_array_equal(frame[f"{OBS_STR}.state"], expected_state)
    np.testing.assert_array_equal(obs["motor_currents"], expected_currents)
    np.testing.assert_array_equal(frame[OBS_MOTOR_CURRENTS], expected_currents)
    follower.bus.sync_read.assert_any_call("Present_Current", num_retry=3, normalize=False)


def test_get_observation_includes_raw_motor_velocity_when_enabled(follower):
    follower.config.observe_motor_velocity = True

    follower.connect()
    obs = follower.get_observation()
    dataset_features = hw_to_dataset_features(follower.observation_features, OBS_STR, use_video=True)
    frame = build_dataset_frame(dataset_features, obs, prefix=OBS_STR)

    expected_position_keys = [f"{motor}.pos" for motor in follower.bus.motors]
    expected_velocity_keys = [f"{motor}.velocity" for motor in follower.bus.motors]
    expected_state = np.array([*range(1, len(follower.bus.motors) + 1)], dtype=np.float32)
    expected_velocities = np.array(
        [-idx * 10 for idx in range(1, len(follower.bus.motors) + 1)],
        dtype=np.int32,
    )

    assert set(obs) == {*expected_position_keys, "motor_velocities"}
    assert follower.action_features == dict.fromkeys(expected_position_keys, float)
    assert dataset_features[f"{OBS_STR}.state"]["names"] == expected_position_keys
    assert dataset_features[OBS_MOTOR_VELOCITIES] == {
        "dtype": "int32",
        "shape": (len(follower.bus.motors),),
        "names": expected_velocity_keys,
    }
    np.testing.assert_array_equal(frame[f"{OBS_STR}.state"], expected_state)
    np.testing.assert_array_equal(obs["motor_velocities"], expected_velocities)
    np.testing.assert_array_equal(frame[OBS_MOTOR_VELOCITIES], expected_velocities)
    follower.bus.sync_read.assert_any_call("Present_Velocity", num_retry=3, normalize=False)


def test_get_observation_includes_metric_depth_for_depth_enabled_camera(follower):
    camera_name = "angled_realsense"
    rgb_image = np.full((2, 3, 3), 7, dtype=np.uint8)
    depth_map = np.array([[0, 2500, 5000], [6000, 1000, 4000]], dtype=np.uint16)

    camera = MagicMock(name="DepthCamera")
    camera.is_connected = True
    camera.read_latest_rgbd.return_value = (rgb_image, depth_map)

    follower.config.cameras = {camera_name: SimpleNamespace(height=2, width=3, use_depth=True)}
    follower.cameras = {camera_name: camera}

    raw_depth_key = f"depths.{camera_name}"
    dataset_depth_key = f"{OBS_DEPTHS}.{camera_name}"
    assert follower.observation_features[camera_name] == (2, 3, 3)
    assert follower.observation_features[raw_depth_key] == {
        "dtype": "uint16",
        "shape": (2, 3),
        "names": ["height", "width"],
    }

    follower.connect()
    obs = follower.get_observation()
    dataset_features = hw_to_dataset_features(follower.observation_features, OBS_STR, use_video=True)
    frame = build_dataset_frame(dataset_features, obs, prefix=OBS_STR)

    np.testing.assert_array_equal(obs[camera_name], rgb_image)
    np.testing.assert_array_equal(obs[raw_depth_key], depth_map)
    assert obs[raw_depth_key].dtype == np.uint16
    assert dataset_features[dataset_depth_key] == {
        "dtype": "uint16",
        "shape": (2, 3),
        "names": ["height", "width"],
    }
    np.testing.assert_array_equal(frame[dataset_depth_key], depth_map)
    np.testing.assert_array_equal(frame[f"{OBS_IMAGES}.{camera_name}"], rgb_image)
    camera.read_latest_rgbd.assert_called_once_with()
    camera.read_latest.assert_not_called()
    camera.read_depth.assert_not_called()


def test_send_action(follower):
    follower.connect()

    action = {f"{m}.pos": i * 10 for i, m in enumerate(follower.bus.motors, 1)}
    returned = follower.send_action(action)

    assert returned == action

    goal_pos = {m: (i + 1) * 10 for i, m in enumerate(follower.bus.motors)}
    follower.bus.sync_write.assert_called_once_with("Goal_Position", goal_pos)
