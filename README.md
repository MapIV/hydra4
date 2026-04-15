# hydra4

A Python library for decoding [Hesai](https://www.hesaitech.com/) Pandar LiDAR packets into [pypcd4](https://github.com/MapIV/pypcd4) `PointCloud` objects.

hydra4 converts raw `pandar_msgs/PandarScan` messages — whether read from a rosbag file or received in real time over ROS 2 — into structured point clouds that are ready for downstream processing.

## Supported Models

| Model | Channels | Default Range | Return Modes |
| -------------- | -------- | ------------- | --------------------------------- |
| Pandar XT-32 | 32 | 0.1 – 120.0 m | Single / Dual (Last-Strongest, …) |
| Pandar OT-128 | 128 | 0.3 – 230.0 m | Single / Dual, High-Res & Standard |

## Requirements

- Python ≥ 3.10
- [numpy](https://numpy.org/) ≥ 1.21.0
- [pypcd4](https://github.com/MapIV/pypcd4) ≥ 1.0.0

## Installation

```bash
pip install hydra4
```

## Quick Start

```python
from hydra4 import Hydra

hydra = Hydra(model=Hydra.Model.PandarXT32)

# `pandar_scan_msg` is a pandar_msgs/PandarScan message
point_clouds = hydra.to_pypcd4(pandar_scan_msg)
for pc in point_clouds:
    print(pc.pc_data["x"][:5])
```

## Usage with rosbags

[rosbags](https://github.com/rpng/rosbags) can read ROS 1 and ROS 2 bag files without a ROS installation.
The example below decodes every `PandarScan` message in a bag and converts them to `pypcd4.PointCloud` objects.

```python
from pathlib import Path

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

from hydra4 import Hydra

# 1. Register the Pandar message types with rosbags
typestore = get_typestore(Stores.LATEST)
typestore.register(
    get_types_from_msg(
        "builtin_interfaces/Time stamp\nuint8[1500] data\nuint32 size\n",
        "pandar_msgs/msg/PandarPacket",
    )
)
typestore.register(
    get_types_from_msg(
        "std_msgs/Header header\nPandarPacket[] packets\n",
        "pandar_msgs/msg/PandarScan",
    )
)

# 2. Create the decoder
hydra = Hydra(model=Hydra.Model.PandarXT32)

# 3. Read the bag and decode each scan
bag_path = Path("path/to/rosbag")
with AnyReader([bag_path], default_typestore=typestore) as reader:
    for connection, _timestamp, rawdata in reader.messages():
        msg = reader.deserialize(rawdata, connection.msgtype)

        for pc in hydra.to_pypcd4(msg):
            # Each `pc` is a pypcd4.PointCloud
            print(f"Frame with {pc.points} points")
            # Save to PCD file
            pc.save("output.pcd")
```

## Usage with a ROS 2 Subscriber

You can also use hydra4 inside a standard ROS 2 subscriber node.

```python
import rclpy
from rclpy.node import Node

from hydra4 import Hydra


class PandarDecoderNode(Node):
    def __init__(self) -> None:
        super().__init__("pandar_decoder_node")
        self.hydra = Hydra(model=Hydra.Model.PandarXT32)

        self.subscription = self.create_subscription(
            msg_type=...,  # pandar_msgs.msg.PandarScan
            topic="/pandar/pandar_packets",
            callback=self.on_scan,
            qos_profile=10,
        )

    def on_scan(self, msg) -> None:
        for pc in self.hydra.to_pypcd4(msg):
            self.get_logger().info(f"Decoded frame: {pc.points} points")
            # Process or publish the point cloud ...


def main() -> None:
    rclpy.init()
    node = PandarDecoderNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
```

## Constructor Parameters

| Parameter | Type | Description |
| --------------------------------- | --------------------- | ------------------------------------------------------------- |
| `model` | `Hydra.Model` | LiDAR model (`PandarXT32` or `PandarOT128`) |
| `calibration_path` | `str \| Path \| None` | Path to a calibration CSV. Defaults to the built-in file. |
| `min_distance` | `float \| None` | Minimum valid distance in metres. Defaults to model spec. |
| `max_distance` | `float \| None` | Maximum valid distance in metres. Defaults to model spec. |
| `dual_return_distance_threshold` | `float` | Dual-return deduplication threshold in metres (default: 0.1). |

### Dual-Return Deduplication

When the sensor operates in a dual-return mode (e.g. Last-Strongest), two blocks share the same azimuth angle. If the distance difference between the first and last return for the same channel is smaller than `dual_return_distance_threshold`, the first return is discarded to avoid near-duplicate points. Set the threshold to `0.0` to keep all returns.

## Output Fields

Each `pypcd4.PointCloud` contains a structured NumPy array with the following fields:

| Field | Type | Description |
| ----------- | ------- | ---------------------------------- |
| `x` | float32 | X coordinate in metres |
| `y` | float32 | Y coordinate in metres |
| `z` | float32 | Z coordinate in metres |
| `intensity` | float32 | Return intensity (0 – 255) |
| `ring` | uint16 | Channel / laser ring index |
| `azimuth` | float32 | Raw azimuth value (hundredths deg) |
| `stamp` | float64 | Per-point timestamp (Unix epoch) |

## Project Structure

```text
src/hydra4/
├── __init__.py          # Public API — exports Hydra
├── hydra4.py            # Hydra class (message → PointCloud pipeline)
├── structs.py           # ReturnMode enum, Block dataclass
├── calibrations/        # Built-in calibration CSVs
│   ├── pandar_xt32.csv
│   └── pandar_ot128.csv
├── decoders/
│   ├── base.py          # PandarBase abstract decoder
│   ├── pandar_xt32.py   # XT-32 packet decoder
│   └── pandar_ot128.py  # OT-128 packet decoder
└── msgs/                # Lightweight ROS message dataclasses
    ├── pandar_msgs.py
    ├── std_msgs.py
    └── builtin_interfaces.py
```

## Development

```bash
# Set up the development environment
uv sync --group dev

# Run the full test suite
pytest tests -vv

# Run only unit / integration / e2e tests
pytest tests/unit -vv
pytest tests/integration -vv
pytest tests/e2e -vv

# Lint and format
ruff check src/ tests/
ruff format src/ tests/
```

## License

Apache License 2.0 — see [LICENSE.txt](LICENSE.txt) for details.
