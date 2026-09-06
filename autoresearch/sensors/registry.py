"""Part registry + vehicle suite builder."""
from .specs.mpu9250 import MPU9250_SPEC
from .imu import SimIMU

SPECS = {"mpu9250": MPU9250_SPEC}
SENSOR_CLASSES = {"imu": SimIMU}

def build_suite(vehicle_cfg, seed=0):
    """vehicle_cfg: [{"type": "imu", "part": "mpu9250", "mount": {...}, "name": "fc_imu"}]"""
    from .base import Mount
    import numpy as np
    suite = []
    for i, entry in enumerate(vehicle_cfg):
        spec = SPECS[entry["part"]]
        cls = SENSOR_CLASSES[entry["type"]]
        m = entry.get("mount", {})
        mount = Mount(pos=np.array(m.get("pos", [0, 0, 0]), dtype=float),
                      rot=np.array(m.get("rot", np.eye(3).tolist()), dtype=float))
        suite.append(cls(spec, mount=mount, seed=seed + i, name=entry.get("name")))
    return suite
