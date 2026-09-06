"""MPU-9250 datasheet anchors (PS-MPU-9250A-01 v1.1, TDK/InvenSense).
Source: https://media.digikey.com/pdf/Data%20Sheets/TDK%20PDFs/MPU-9250_Rev_1.1.pdf
LITERATURE-CLASS (not datasheet): OU bias drift rates, accel ZGO, scale_tol.
"""
MPU9250_SPEC = dict(
    part="mpu9250",
    rate_hz=500.0,               # physics-bound output; part supports 8k gyro/4k accel
    gyro_fs_dps=2000.0,
    accel_fs_g=16.0,
    gyro_rms_92hz=0.1,           # dps-rms @ DLPFCFG=2 (datasheet anchor)
    gyro_nd_dps=0.01,            # dps/sqrt(Hz) typ
    accel_nd_ug=300.0,           # ug/sqrt(Hz) typ
    gyro_zro_init_dps=5.0,
    gyro_zro_temp_span_dps=30.0, # -40..85C envelope, centered 25C
    accel_zgo_init_mg=60.0,      # LITERATURE-CLASS; pin exact from PDF table
    cross_axis=0.02,
    scale_tol=0.01,              # LITERATURE-CLASS
    dlpf_hz=92.0,
    bits=16,
    gyro_ou_sigma_dps=1.0, gyro_ou_tau_s=300.0,      # LITERATURE-CLASS
    accel_ou_sigma_mg=0.5, accel_ou_tau_s=300.0,     # LITERATURE-CLASS
)
