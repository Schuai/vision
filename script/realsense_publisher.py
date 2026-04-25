#!/usr/bin/env python3
"""Compatibility shim for the ROS2 tracking package."""

from __future__ import annotations

import sys
from pathlib import Path


ROS_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "ros2" / "tracking"
if str(ROS_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(ROS_PACKAGE_ROOT))

from tracking.realsense_publisher import *  # noqa: F401,F403,E402
from tracking.realsense_publisher import main  # noqa: E402


if __name__ == "__main__":
    main()
