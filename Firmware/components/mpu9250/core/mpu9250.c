#include "mpu9250.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "MPU9250";

// Static variables
static i2c_port_t i2c_port;
static float accel_scale = 16384.0f;  // ±2g default
static float gyro_scale = 131.0f;     // ±250°/s default
static float mag_scale[3] = {0};

// Helper functions for I2C communication
static esp_err_t write_byte(uint8_t addr, uint8_t reg, uint8_t data)
{
    i2c_cmd_handle_t cmd = i2c_cmd_link_create();
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (addr << 1) | I2C_MASTER_WRITE, true);
    i2c_master_write_byte(cmd, reg, true);
    i2c_master_write_byte(cmd, data, true);
    i2c_master_stop(cmd);
    esp_err_t ret = i2c_master_cmd_begin(i2c_port, cmd, pdMS_TO_TICKS(100));
    i2c_cmd_link_delete(cmd);
    return ret;
}

static esp_err_t read_byte(uint8_t addr, uint8_t reg, uint8_t *data)
{
    i2c_cmd_handle_t cmd = i2c_cmd_link_create();
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (addr << 1) | I2C_MASTER_WRITE, true);
    i2c_master_write_byte(cmd, reg, true);
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (addr << 1) | I2C_MASTER_READ, true);
    i2c_master_read_byte(cmd, data, I2C_MASTER_NACK);
    i2c_master_stop(cmd);
    esp_err_t ret = i2c_master_cmd_begin(i2c_port, cmd, pdMS_TO_TICKS(100));
    i2c_cmd_link_delete(cmd);
    return ret;
}

static esp_err_t read_bytes(uint8_t addr, uint8_t reg, uint8_t *data, size_t len)
{
    i2c_cmd_handle_t cmd = i2c_cmd_link_create();
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (addr << 1) | I2C_MASTER_WRITE, true);
    i2c_master_write_byte(cmd, reg, true);
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (addr << 1) | I2C_MASTER_READ, true);
    if (len > 1) {
        i2c_master_read(cmd, data, len - 1, I2C_MASTER_ACK);
    }
    i2c_master_read_byte(cmd, data + len - 1, I2C_MASTER_NACK);
    i2c_master_stop(cmd);
    esp_err_t ret = i2c_master_cmd_begin(i2c_port, cmd, pdMS_TO_TICKS(100));
    i2c_cmd_link_delete(cmd);
    return ret;
}

esp_err_t mpu9250_init(mpu9250_config_t *config)
{
    esp_err_t ret;
    uint8_t data;

    // Store I2C port
    i2c_port = config->i2c_port;

    // Configure I2C
    i2c_config_t i2c_conf = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = config->sda_pin,
        .scl_io_num = config->scl_pin,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master.clk_speed = config->clk_speed
    };

    ret = i2c_param_config(i2c_port, &i2c_conf);
    if (ret != ESP_OK) return ret;

    ret = i2c_driver_install(i2c_port, I2C_MODE_MASTER, 0, 0, 0);
    if (ret != ESP_OK) return ret;

    // Check MPU9250 WHO_AM_I
    ret = read_byte(MPU9250_ADDR, WHO_AM_I, &data);
    if (ret != ESP_OK) return ret;
    if (data != 0x71) {
        ESP_LOGE(TAG, "MPU9250 WHO_AM_I check failed: %02x", data);
        return ESP_ERR_NOT_FOUND;
    }

    // Initialize MPU9250
    ret = write_byte(MPU9250_ADDR, PWR_MGMT_1, 0x00);  // Wake up
    if (ret != ESP_OK) return ret;
    vTaskDelay(pdMS_TO_TICKS(100));

    ret = write_byte(MPU9250_ADDR, ACCEL_CONFIG, 0x00);  // ±2g range
    if (ret != ESP_OK) return ret;

    ret = write_byte(MPU9250_ADDR, GYRO_CONFIG, 0x00);   // ±250°/s range
    if (ret != ESP_OK) return ret;

    // Enable I2C bypass to access magnetometer
    ret = write_byte(MPU9250_ADDR, INT_PIN_CFG, 0x02);
    if (ret != ESP_OK) return ret;
    vTaskDelay(pdMS_TO_TICKS(10));

    // Initialize magnetometer
    ret = write_byte(AK8963_ADDR, AK8963_CNTL1, 0x00);  // Power down
    if (ret != ESP_OK) return ret;
    vTaskDelay(pdMS_TO_TICKS(10));

    ret = write_byte(AK8963_ADDR, AK8963_CNTL1, 0x0F);  // Fuse ROM access
    if (ret != ESP_OK) return ret;
    vTaskDelay(pdMS_TO_TICKS(10));

    // Read magnetometer sensitivity adjustment values
    uint8_t raw_data[3];
    ret = read_bytes(AK8963_ADDR, AK8963_ASAX, raw_data, 3);
    if (ret != ESP_OK) return ret;

    mag_scale[0] = (float)(raw_data[0] - 128) / 256.0f + 1.0f;
    mag_scale[1] = (float)(raw_data[1] - 128) / 256.0f + 1.0f;
    mag_scale[2] = (float)(raw_data[2] - 128) / 256.0f + 1.0f;

    // Set magnetometer to continuous measurement mode (16-bit, 100Hz)
    ret = write_byte(AK8963_ADDR, AK8963_CNTL1, 0x16);
    if (ret != ESP_OK) return ret;
    vTaskDelay(pdMS_TO_TICKS(10));

    ESP_LOGI(TAG, "MPU9250 initialized successfully");
    return ESP_OK;
}

esp_err_t mpu9250_read_sensors(mpu9250_data_t *data)
{
    esp_err_t ret;
    uint8_t raw_data[6];
    int16_t raw_value[3];

    // Read accelerometer
    ret = read_bytes(MPU9250_ADDR, ACCEL_XOUT_H, raw_data, 6);
    if (ret != ESP_OK) return ret;

    raw_value[0] = (int16_t)(((int16_t)raw_data[0] << 8) | raw_data[1]);
    raw_value[1] = (int16_t)(((int16_t)raw_data[2] << 8) | raw_data[3]);
    raw_value[2] = (int16_t)(((int16_t)raw_data[4] << 8) | raw_data[5]);

    data->accel_x = (float)raw_value[0] / accel_scale;
    data->accel_y = (float)raw_value[1] / accel_scale;
    data->accel_z = (float)raw_value[2] / accel_scale;

    // Read gyroscope
    ret = read_bytes(MPU9250_ADDR, GYRO_XOUT_H, raw_data, 6);
    if (ret != ESP_OK) return ret;

    raw_value[0] = (int16_t)(((int16_t)raw_data[0] << 8) | raw_data[1]);
    raw_value[1] = (int16_t)(((int16_t)raw_data[2] << 8) | raw_data[3]);
    raw_value[2] = (int16_t)(((int16_t)raw_data[4] << 8) | raw_data[5]);

    data->gyro_x = (float)raw_value[0] / gyro_scale;
    data->gyro_y = (float)raw_value[1] / gyro_scale;
    data->gyro_z = (float)raw_value[2] / gyro_scale;

    // Read magnetometer
    uint8_t st1;
    ret = read_byte(AK8963_ADDR, AK8963_ST1, &st1);
    if (ret != ESP_OK) return ret;

    if (st1 & 0x01) {
        ret = read_bytes(AK8963_ADDR, AK8963_XOUT_L, raw_data, 6);
        if (ret != ESP_OK) return ret;

        uint8_t st2;
        ret = read_byte(AK8963_ADDR, AK8963_ST2, &st2);
        if (ret != ESP_OK) return ret;

        if (!(st2 & 0x08)) {
            raw_value[0] = (int16_t)(((int16_t)raw_data[1] << 8) | raw_data[0]);
            raw_value[1] = (int16_t)(((int16_t)raw_data[3] << 8) | raw_data[2]);
            raw_value[2] = (int16_t)(((int16_t)raw_data[5] << 8) | raw_data[4]);

            data->mag_x = (float)raw_value[0] * mag_scale[0] * 0.15f;
            data->mag_y = (float)raw_value[1] * mag_scale[1] * 0.15f;
            data->mag_z = (float)raw_value[2] * mag_scale[2] * 0.15f;
        }
    }

    return ESP_OK;
}