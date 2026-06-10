#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import logging
import time
from functools import cached_property

import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.constants import (
    OBS_DEPTHS,
    OBS_MOTOR_CURRENTS,
    OBS_MOTOR_VELOCITIES,
    OBS_SENSOR_TIMESTAMPS,
    OBS_STR,
)
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_so_follower import SOFollowerRobotConfig

logger = logging.getLogger(__name__)
RAW_DEPTHS = OBS_DEPTHS.removeprefix(f"{OBS_STR}.")
RAW_MOTOR_CURRENTS = OBS_MOTOR_CURRENTS.removeprefix(f"{OBS_STR}.")
RAW_MOTOR_VELOCITIES = OBS_MOTOR_VELOCITIES.removeprefix(f"{OBS_STR}.")
RAW_SENSOR_TIMESTAMPS = OBS_SENSOR_TIMESTAMPS.removeprefix(f"{OBS_STR}.")


class SOFollower(Robot):
    """
    Generic SO follower base implementing common functionality for SO-100/101/10X.
    Designed to be subclassed with a per-hardware-model `config_class` and `name`.
    """

    config_class = SOFollowerRobotConfig
    name = "so_follower"

    def __init__(self, config: SOFollowerRobotConfig):
        super().__init__(config)
        self.config = config
        # choose normalization mode depending on config if available
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def _motor_currents_ft(self) -> dict[str, dict]:
        if not self.config.observe_motor_current:
            return {}
        return {
            RAW_MOTOR_CURRENTS: {
                "dtype": "int32",
                "shape": (len(self.bus.motors),),
                "names": [f"{motor}.current" for motor in self.bus.motors],
            }
        }

    @property
    def _motor_velocities_ft(self) -> dict[str, dict]:
        if not self.config.observe_motor_velocity:
            return {}
        return {
            RAW_MOTOR_VELOCITIES: {
                "dtype": "int32",
                "shape": (len(self.bus.motors),),
                "names": [f"{motor}.velocity" for motor in self.bus.motors],
            }
        }

    @property
    def _sensor_timestamp_names(self) -> list[str]:
        # Keep this order identical to the order timestamps are appended in `get_observation`.
        names = ["motor_positions.perf_counter_s"]
        if self.config.observe_motor_current:
            names.append("motor_currents.perf_counter_s")
        if self.config.observe_motor_velocity:
            names.append("motor_velocities.perf_counter_s")
        names.extend(f"camera.{cam}.perf_counter_s" for cam in self.cameras)
        return names

    @property
    def _sensor_timestamps_ft(self) -> dict[str, dict]:
        if not self.config.observe_sensor_timestamps:
            return {}
        names = self._sensor_timestamp_names
        return {
            RAW_SENSOR_TIMESTAMPS: {
                # One vector per dataset frame. `names` identifies each timestamp column.
                "dtype": "float64",
                "shape": (len(names),),
                "names": names,
            }
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple | dict]:
        camera_features = {}
        for cam in self.cameras:
            camera_features[cam] = (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            if getattr(self.config.cameras[cam], "use_depth", False):
                camera_features[f"{RAW_DEPTHS}.{cam}"] = {
                    "dtype": "uint16",
                    "shape": (self.config.cameras[cam].height, self.config.cameras[cam].width),
                    "names": ["height", "width"],
                }
        return camera_features

    @cached_property
    def observation_features(self) -> dict[str, type | tuple | dict]:
        return {
            **self._motors_ft,
            **self._motor_currents_ft,
            **self._motor_velocities_ft,
            **self._sensor_timestamps_ft,
            **self._cameras_ft,
        }

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """
        We assume that at connection time, arm is in a rest position,
        and torque can be safely disabled to run calibration.
        """

        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        if self.calibration:
            # Calibration file exists, ask user whether to use it or run new calibration
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                return

        logger.info(f"\nRunning calibration of {self}")
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input(f"Move {self} to the middle of its range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings()

        # Attempt to call record_ranges_of_motion with a reduced motor set when appropriate.
        full_turn_motor = "wrist_roll"
        unknown_range_motors = [motor for motor in self.bus.motors if motor != full_turn_motor]
        print(
            f"Move all joints except '{full_turn_motor}' sequentially through their "
            "entire ranges of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion(unknown_range_motors)
        range_mins[full_turn_motor] = 0
        range_maxes[full_turn_motor] = 4095

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)

    def configure(self) -> None:
        with self.bus.torque_disabled():
            self.bus.configure_motors()
            for motor in self.bus.motors:
                self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
                # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
                self.bus.write("P_Coefficient", motor, 16)
                # Set I_Coefficient and D_Coefficient to default value 0 and 32
                self.bus.write("I_Coefficient", motor, 0)
                self.bus.write("D_Coefficient", motor, 32)

                if motor == "gripper":
                    self.bus.write("Max_Torque_Limit", motor, 500)  # 50% of max torque to avoid burnout
                    self.bus.write("Protection_Current", motor, 250)  # 50% of max current to avoid burnout
                    self.bus.write("Overload_Torque", motor, 25)  # 25% torque when overloaded

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        # These timestamps use Python's monotonic clock (`time.perf_counter`) so they are comparable
        # within one recording process. They are not wall-clock time and should only be used via differences.
        sensor_timestamps = []

        # Read arm position
        start = time.perf_counter()
        obs_dict = self.bus.sync_read("Present_Position", num_retry=3)
        read_end = time.perf_counter()
        if self.config.observe_sensor_timestamps:
            # Motor registers are read with blocking serial transactions, so the exact sample instant is
            # unknown. The midpoint is the best software estimate using the same clock as the cameras.
            sensor_timestamps.append((start + read_end) / 2)
        obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}
        dt_ms = (read_end - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Read arm motor currents when enabled in config. Can be used for force estimation and collision detection.
        if self.config.observe_motor_current:
            start = time.perf_counter()
            current_values = self.bus.sync_read("Present_Current", num_retry=3, normalize=False)
            read_end = time.perf_counter()
            if self.config.observe_sensor_timestamps:
                # Same midpoint convention as position: one timestamp for the whole multi-motor sync read.
                sensor_timestamps.append((start + read_end) / 2)
            obs_dict[RAW_MOTOR_CURRENTS] = np.array(
                [current_values[motor] for motor in self.bus.motors],
                dtype=np.int32,
            )
            dt_ms = (read_end - start) * 1e3
            logger.debug(f"{self} read motor currents: {dt_ms:.1f}ms")

        # Read arm motor velocities when enabled in config. Can be used for velocity-based control or
        # dynamics estimation.
        if self.config.observe_motor_velocity:
            start = time.perf_counter()
            velocity_values = self.bus.sync_read("Present_Velocity", num_retry=3, normalize=False)
            read_end = time.perf_counter()
            if self.config.observe_sensor_timestamps:
                # Same midpoint convention as position: one timestamp for the whole multi-motor sync read.
                sensor_timestamps.append((start + read_end) / 2)
            obs_dict[RAW_MOTOR_VELOCITIES] = np.array(
                [velocity_values[motor] for motor in self.bus.motors],
                dtype=np.int32,
            )
            dt_ms = (read_end - start) * 1e3
            logger.debug(f"{self} read motor velocities: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            # Default to RGB-only for cameras that do not define `use_depth` (e.g. OpenCV cameras).
            # If this default were True, non-depth cameras would incorrectly enter the RGB-D path.
            if getattr(self.config.cameras[cam_key], "use_depth", False):
                # For depth-enabled RealSense cameras, read RGB and depth together from the same
                # cached frameset so dataset frames do not mix RGB from one timestamp with depth from another.
                read_latest_rgbd = getattr(cam, "read_latest_rgbd", None)
                if read_latest_rgbd is None:
                    raise RuntimeError(
                        f"Camera '{cam_key}' has use_depth=True but does not support "
                        "synchronized RGB-D reads."
                    )
                if self.config.observe_sensor_timestamps:
                    read_latest_rgbd_with_timestamp = getattr(cam, "read_latest_rgbd_with_timestamp", None)
                    if read_latest_rgbd_with_timestamp is not None:
                        # RealSense RGB and depth share this cached software timestamp because they come
                        # from the same frameset in the camera read thread.
                        obs_dict[cam_key], obs_dict[f"{RAW_DEPTHS}.{cam_key}"], camera_timestamp = (
                            read_latest_rgbd_with_timestamp()
                        )
                    else:
                        # Fallback for any future depth camera that lacks timestamp support.
                        obs_dict[cam_key], obs_dict[f"{RAW_DEPTHS}.{cam_key}"] = read_latest_rgbd()
                        camera_timestamp = (start + time.perf_counter()) / 2
                    sensor_timestamps.append(camera_timestamp)
                else:
                    obs_dict[cam_key], obs_dict[f"{RAW_DEPTHS}.{cam_key}"] = read_latest_rgbd()
            else:
                obs_dict[cam_key] = cam.read_latest()
                if self.config.observe_sensor_timestamps:
                    # Generic cameras currently expose no hardware/cached frame timestamp, so use the
                    # midpoint of the software read call.
                    sensor_timestamps.append((start + time.perf_counter()) / 2)
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        if self.config.observe_sensor_timestamps:
            obs_dict[RAW_SENSOR_TIMESTAMPS] = np.array(sensor_timestamps, dtype=np.float64)

        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """Command arm to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Raises:
            RobotDeviceNotConnectedError: if robot is not connected.

        Returns:
            RobotAction: the action sent to the motors, potentially clipped.
        """

        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        # Cap goal position when too far away from present position.
        # /!\ Slower fps expected due to reading from the follower.
        if self.config.max_relative_target is not None:
            present_pos = self.bus.sync_read("Present_Position")
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        # Send goal position to the arm
        self.bus.sync_write("Goal_Position", goal_pos)
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    @check_if_not_connected
    def disconnect(self):
        self.bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")


SO100Follower = SOFollower
SO101Follower = SOFollower
