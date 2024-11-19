idf_component_register(
    SRCS "core/mpu9250.c"
    INCLUDE_DIRS "include"
    REQUIRES nvs_flash
    REQUIRES driver
)