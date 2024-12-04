#include "mpu9250.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <stdint.h>
#include <inttypes.h>
#include "nvs_flash.h"
#include "nvs.h"

static const char *TAG = "MPU9250";

// Static variables
static i2c_port_t i2c_port;
static float accel_scale = 4096.0f;  // ±8g default
static float gyro_scale = 131.0f;     // ±250°/s default
static float mag_scale[3] = {0};

// Calibration variables
static float mag_offset[3] = {0};
static float mag_scale_factor[3] = {1.0f, 1.0f, 1.0f};


// NVS namespace and keys
#define NVS_NAMESPACE "calibration"
#define MAG_OFFSET_KEY "mag_offset"
#define MAG_SCALE_KEY "mag_scale"


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

// NVS Initialization
static esp_err_t init_nvs()
{
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    return err;
}

// Save calibration data to NVS
static esp_err_t save_calibration()
{
    nvs_handle_t handle;
    esp_err_t err = nvs_open(NVS_NAMESPACE, NVS_READWRITE, &handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to open NVS namespace for writing");
        return err;
    }

    err = nvs_set_blob(handle, MAG_OFFSET_KEY, mag_offset, sizeof(mag_offset));
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to set mag_offset in NVS");
        nvs_close(handle);
        return err;
    }

    err = nvs_set_blob(handle, MAG_SCALE_KEY, mag_scale_factor, sizeof(mag_scale_factor));
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to set mag_scale in NVS");
        nvs_close(handle);
        return err;
    }

    err = nvs_commit(handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to commit calibration data to NVS");
    }

    nvs_close(handle);
    return err;
}


// Calibration function for magnetometer
esp_err_t mpu9250_calibrate_magnetometer(uint32_t sample_count, uint32_t delay_ms)
{
    ESP_LOGI(TAG, "Starting magnetometer calibration...");
    ESP_LOGI(TAG, "Please move in a figure 8 on a flat surface until timer ends.");

    float mag_min[3] = {INT16_MAX, INT16_MAX, INT16_MAX};
    float mag_max[3] = {INT16_MIN, INT16_MIN, INT16_MIN};

    for (uint32_t i = 0; i < sample_count; i++) {
        mpu9250_data_t data;
        esp_err_t ret = mpu9250_read_sensors(&data);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "Failed to read sensors during calibration");
            return ret;
        }

        printf("Time Remaining: %" PRIu32 "ms", (sample_count - i) * 10);

        // Update min and max
        if (data.mag_x < mag_min[0]) mag_min[0] = data.mag_x;
        if (data.mag_x > mag_max[0]) mag_max[0] = data.mag_x;

        if (data.mag_y < mag_min[1]) mag_min[1] = data.mag_y;
        if (data.mag_y > mag_max[1]) mag_max[1] = data.mag_y;

        if (data.mag_z < mag_min[2]) mag_min[2] = data.mag_z;
        if (data.mag_z > mag_max[2]) mag_max[2] = data.mag_z;

        vTaskDelay(pdMS_TO_TICKS(delay_ms));
    }

    // Calculate offsets (hard iron)
    mag_offset[0] = (mag_max[0] + mag_min[0]) / 2.0f;
    mag_offset[1] = (mag_max[1] + mag_min[1]) / 2.0f;
    mag_offset[2] = (mag_max[2] + mag_min[2]) / 2.0f;

    // Calculate scale factors (soft iron)
    float mag_range_x = mag_max[0] - mag_min[0];
    float mag_range_y = mag_max[1] - mag_min[1];
    float mag_range_z = mag_max[2] - mag_min[2];
    float avg_range = (mag_range_x + mag_range_y + mag_range_z) / 3.0f;

    mag_scale_factor[0] = avg_range / mag_range_x;
    mag_scale_factor[1] = avg_range / mag_range_y;
    mag_scale_factor[2] = avg_range / mag_range_z;

    ESP_LOGI(TAG, "Calibration complete:");
  
    // Save calibration data to NVS
    esp_err_t err = save_calibration();
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to save calibration data");
        return err;
    }

    return ESP_OK;
}


// Load calibration data from NVS
static esp_err_t load_calibration()
{
    nvs_handle_t handle;
    esp_err_t err = nvs_open(NVS_NAMESPACE, NVS_READONLY, &handle);
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "No calibration data found in NVS");
       return err;
    }

    size_t required_size = sizeof(mag_offset);
    err = nvs_get_blob(handle, MAG_OFFSET_KEY, mag_offset, &required_size);
    if (err != ESP_OK || required_size != sizeof(mag_offset)) {
        ESP_LOGW(TAG, "Failed to load mag_offset from NVS");
    }

    required_size = sizeof(mag_scale_factor);
    err = nvs_get_blob(handle, MAG_SCALE_KEY, mag_scale_factor, &required_size);
    if (err != ESP_OK || required_size != sizeof(mag_scale_factor)) {
        ESP_LOGW(TAG, "Failed to load mag_scale from NVS");
    }

    ESP_LOGI(TAG, "Mag Offsets - X: %.2f, Y: %.2f, Z: %.2f", mag_offset[0], mag_offset[1], mag_offset[2]);
    ESP_LOGI(TAG, "Mag Scale Factors - X: %.2f, Y: %.2f, Z: %.2f", mag_scale_factor[0], mag_scale_factor[1], mag_scale_factor[2]);

    nvs_close(handle);
    return ESP_OK;
}



esp_err_t mpu9250_init(mpu9250_config_t *config)
{
    esp_err_t ret;
    uint8_t data;

    // Initialize NVS
    ret = init_nvs();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize NVS");
        return ret;
    }

    // Load existing calibration data
    esp_err_t calibration = load_calibration();
    if (calibration != ESP_OK) {
        ESP_LOGW(TAG, "Using default calibration values");
    }

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
     if (ret != ESP_OK) {
          ESP_LOGE(TAG, "Failed to initialize i2c in mpu");
        return ret;
    };

    ret = i2c_driver_install(i2c_port, I2C_MODE_MASTER, 0, 0, 0);
     if (ret != ESP_OK) {
          ESP_LOGE(TAG, "Failed to install i2c mpu driver");
        return ret;
    };

    // Check MPU9250 WHO_AM_I
    ret = read_byte(MPU9250_ADDR, WHO_AM_I, &data);
     if (ret != ESP_OK) {
          ESP_LOGE(TAG, "Failed to read WHO_AM_I %d", ret);
        return ret;
    };
    if (data != 0x71) {
        ESP_LOGE(TAG, "MPU9250 WHO_AM_I check failed: %02x", data);
        return ESP_ERR_NOT_FOUND;
    }

    // Initialize MPU9250
    ret = write_byte(MPU9250_ADDR, PWR_MGMT_1, 0x00);  // Wake up
    if (ret != ESP_OK) {
          ESP_LOGE(TAG, "Failed to wake up");
        return ret;
    }
    vTaskDelay(pdMS_TO_TICKS(100));

    ret = write_byte(MPU9250_ADDR, ACCEL_CONFIG, 0x10);  // ±8g range
    if (ret != ESP_OK) {
          ESP_LOGE(TAG, "Failed to set accel config");
        return ret;
    }

    ret = write_byte(MPU9250_ADDR, GYRO_CONFIG, 0x00);   // ±250°/s range
    if (ret != ESP_OK) {
          ESP_LOGE(TAG, "Failed to set gyro config");
        return ret;
    }

    // Enable I2C bypass to access magnetometer
    ret = write_byte(MPU9250_ADDR, INT_PIN_CFG, 0x02);
    if (ret != ESP_OK) {
          ESP_LOGE(TAG, "Failed to bypass i2c for mag");
        return ret;
    }
    vTaskDelay(pdMS_TO_TICKS(10));

    // Initialize magnetometer
    ret = write_byte(AK8963_ADDR, AK8963_CNTL1, 0x00);  // Power down
    if (ret != ESP_OK) {
          ESP_LOGE(TAG, "Failed to reset mag");
        return ret;
    }
    vTaskDelay(pdMS_TO_TICKS(10));

    ret = write_byte(AK8963_ADDR, AK8963_CNTL1, 0x0F);  // Fuse ROM access
    if (ret != ESP_OK) {
          ESP_LOGE(TAG, "Failed to fuse mag ROM");
        return ret;
    }
    vTaskDelay(pdMS_TO_TICKS(10));

    // Read magnetometer sensitivity adjustment values
    uint8_t raw_data[3];
    ret = read_bytes(AK8963_ADDR, AK8963_ASAX, raw_data, 3);
    if (ret != ESP_OK) {
          ESP_LOGE(TAG, "Failed to read mag sensitivity adjustments");
        return ret;
    }

    mag_scale[0] = (float)(raw_data[0] - 128) / 256.0f + 1.0f;
    mag_scale[1] = (float)(raw_data[1] - 128) / 256.0f + 1.0f;
    mag_scale[2] = (float)(raw_data[2] - 128) / 256.0f + 1.0f;

    // Set magnetometer to continuous measurement mode (16-bit, 100Hz)
    ret = write_byte(AK8963_ADDR, AK8963_CNTL1, 0x16);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to set to continous measurement mode");
        return ret;
    }
    vTaskDelay(pdMS_TO_TICKS(10));

    if (calibration != ESP_OK){
        ret = mpu9250_calibrate_magnetometer(6000, 10);

        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "Magnetometer calibration failed");
            return ret;
        }

        load_calibration();
    }

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

            // ESP_LOGI(TAG, "Raw X: %d, Raw Y: %d, Raw Z: %d", raw_value[0], raw_value[1], raw_value[2]);
            // Apply factory calibration
            float mag_x = (float)raw_value[0] * mag_scale[0] * 0.15f;
            float mag_y = (float)raw_value[1] * mag_scale[1] * 0.15f;
            float mag_z = (float)raw_value[2] * mag_scale[2] * 0.15f;

        //    ESP_LOGI(TAG, "Factory scaled X: %.3f, Factory scaled Y: %.3f, Factory scaled Z: %.3f", mag_y, -1.0 * mag_z, mag_x);

            // // Apply hard iron offset
            mag_x -= mag_offset[0];
            mag_y -= mag_offset[1];
            mag_z -= mag_offset[2];

            // Apply soft iron scaling
            mag_x *= mag_scale_factor[0];
            mag_y *= mag_scale_factor[1];
            mag_z *= mag_scale_factor[2];

            // ESP_LOGI(TAG, "Final X: %.3f, Final Y: %.3f, Final Z: %.3f", mag_y, -1.0 * mag_z, mag_x);

            data->mag_x = mag_x;
            data->mag_y = mag_y;
            data->mag_z = mag_z;
        }
    }

    return ESP_OK;
}