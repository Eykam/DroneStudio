#ifndef MPU9250_H
#define MPU9250_H

#include <stdint.h>
#include "driver/i2c.h"
#include "esp_err.h"

// MPU9250 Register Map
#define MPU9250_ADDR         0x68
#define GYRO_CONFIG          0x1B
#define ACCEL_CONFIG         0x1C
#define ACCEL_CONFIG2        0x1D
#define INT_PIN_CFG          0x37
#define ACCEL_XOUT_H        0x3B
#define GYRO_XOUT_H         0x43
#define PWR_MGMT_1          0x6B
#define WHO_AM_I            0x75

// AK8963 (Magnetometer) Registers
#define AK8963_ADDR         0x0C
#define AK8963_WHO_AM_I     0x00
#define AK8963_INFO         0x01
#define AK8963_ST1          0x02
#define AK8963_ST2          0x09
#define AK8963_XOUT_L       0x03
#define AK8963_CNTL1        0x0A
#define AK8963_ASAX         0x10

// Sensor data structure
typedef struct {
    float accel_x;
    float accel_y;
    float accel_z;
    float gyro_x;
    float gyro_y;
    float gyro_z;
    float mag_x;
    float mag_y;
    float mag_z;
} mpu9250_data_t;

// Configuration structure
typedef struct {
    i2c_port_t i2c_port;
    uint8_t sda_pin;
    uint8_t scl_pin;
    uint32_t clk_speed;
} mpu9250_config_t;

// Function declarations
esp_err_t mpu9250_init(mpu9250_config_t *config);
esp_err_t mpu9250_read_sensors(mpu9250_data_t *data);

#endif // MPU9250_H