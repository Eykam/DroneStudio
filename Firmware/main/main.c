#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "mpu9250.h"
#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "lwip/err.h"
#include "lwip/sockets.h"
#include "lwip/sys.h"
#include <lwip/netdb.h>
#include "nvs_flash.h"
#include "nvs.h"


#define ESP_WIFI_SSID      CONFIG_ESP_WIFI_SSID
#define ESP_WIFI_PASS      CONFIG_ESP_WIFI_PASSWORD
#define ESP_MAXIMUM_RETRY  CONFIG_ESP_MAXIMUM_RETRY

#define PORT 8035
#define BUFFER_LENGTH 1500

/* FreeRTOS event group to signal when we are connected*/
static EventGroupHandle_t s_wifi_event_group;
static EventGroupHandle_t event_group;

#define POINT_QUEUE_BIT (1 << 1)

#define CLIENT_IP CONFIG_CLIENT_IP // IP of client you'd like to connect to
#define HOST_IP 

#define WIFI_CONNECTED_BIT BIT0
#define WIFI_FAIL_BIT      BIT1

#define PACKET_SIZE 36  // Bytes

QueueHandle_t point_queue;

static const char *TAG = "MAIN";
static int s_retry_num = 0;

void init_nvs() {
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(err);
    printf("Did err? => (%s)\n", esp_err_to_name(err));
}

static void event_handler(void* arg, esp_event_base_t event_base,
                                int32_t event_id, void* event_data)
{
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        if (s_retry_num < ESP_MAXIMUM_RETRY) {
            esp_wifi_connect();
            s_retry_num++;
            ESP_LOGI(TAG, "retry to connect to the AP");
        } else {
            xEventGroupSetBits(s_wifi_event_group, WIFI_FAIL_BIT);
        }
        ESP_LOGI(TAG,"connect to the AP fail");
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t* event = (ip_event_got_ip_t*) event_data;
        ESP_LOGI(TAG, "got ip:" IPSTR, IP2STR(&event->ip_info.ip));
        s_retry_num = 0;
        xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
    }
}

void wifi_init_sta(void)
{
    s_wifi_event_group = xEventGroupCreate();

    ESP_ERROR_CHECK(esp_netif_init());

    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &event_handler, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &event_handler, NULL));

    wifi_config_t wifi_config = {
        .sta = {
            .ssid = ESP_WIFI_SSID,
            .password = ESP_WIFI_PASS,
	        .threshold.authmode = WIFI_AUTH_WPA2_PSK,

            .pmf_cfg = {
                .capable = true,
                .required = false
            },
        },
    };
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA) );
    ESP_ERROR_CHECK(esp_wifi_set_config(ESP_IF_WIFI_STA, &wifi_config) );
    ESP_ERROR_CHECK(esp_wifi_start() );

    ESP_LOGI(TAG, "wifi_init_sta finished.");

    /* Waiting until either the connection is established (WIFI_CONNECTED_BIT) or connection failed for the maximum
     * number of re-tries (WIFI_FAIL_BIT). The bits are set by event_handler() (see above) */
    EventBits_t bits = xEventGroupWaitBits(s_wifi_event_group,
            WIFI_CONNECTED_BIT | WIFI_FAIL_BIT,
            pdFALSE,
            pdFALSE,
            portMAX_DELAY);

    /* xEventGroupWaitBits() returns the bits before the call returned, hence we can test which event actually
     * happened. */
    if (bits & WIFI_CONNECTED_BIT) {
        ESP_LOGI(TAG, "connected to ap SSID:%s",
                 ESP_WIFI_SSID);
    } else if (bits & WIFI_FAIL_BIT) {
        ESP_LOGI(TAG, "Failed to connect to SSID:%s",
                 ESP_WIFI_SSID);
    } else {
        ESP_LOGE(TAG, "UNEXPECTED EVENT");
    }

    ESP_ERROR_CHECK(esp_event_handler_unregister(IP_EVENT, IP_EVENT_STA_GOT_IP, &event_handler));
    ESP_ERROR_CHECK(esp_event_handler_unregister(WIFI_EVENT, ESP_EVENT_ANY_ID, &event_handler));
    vEventGroupDelete(s_wifi_event_group);
}

void wifi_transmission_task(void *pvParameters) {
    mpu9250_data_t data;
    
    // Create a UDP socket
    int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);

    struct sockaddr_in dest_addr;
    dest_addr.sin_addr.s_addr = inet_addr(CLIENT_IP);
    dest_addr.sin_family = AF_INET;
    dest_addr.sin_port = htons(8035);

    while (1) {
        if (sock < 0) {
            ESP_LOGE(TAG, "Unable to create socket: errno %d", errno);
            sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
            continue;
        }

        // Check both queues without blocking
        EventBits_t bits = xEventGroupWaitBits(
            event_group,
            POINT_QUEUE_BIT,
            pdTRUE,  // Clear bits before returning
            pdFALSE,  // Don't wait for all bits
            0  // Don't block
        );

        if (bits & POINT_QUEUE_BIT) {
            if (xQueueReceive(point_queue, &data, 0) == pdPASS) {
                // uint16_t packet[PACKET_SIZE];
                // Pack data into the packet
                // packet[0] = data;
                
                int err = sendto(sock, &data, sizeof(mpu9250_data_t), 0, (struct sockaddr *)&dest_addr, sizeof(dest_addr));
            
                if (err < 0) {
                    ESP_LOGE(TAG, "Error occurred during sending: errno %d", errno);
                }
            }
        }

        // If neither queue had data, add a small delay to prevent tight-looping
        if ((bits & ( POINT_QUEUE_BIT)) == 0) {
            vTaskDelay(pdMS_TO_TICKS(1));
        }
    }

    shutdown(sock, 0);
    close(sock);
}

void read_sensor_task(void *pvParameters) {
    // Configure MPU9250
    mpu9250_config_t mpu_config = {
        .i2c_port = I2C_NUM_0,
        .sda_pin = GPIO_NUM_26,  // Adjust according to your setup
        .scl_pin = GPIO_NUM_25,  // Adjust according to your setup
        .clk_speed = 400000      // 400 KHz
    };

    // Initialize MPU9250
    esp_err_t ret = mpu9250_init(&mpu_config);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize MPU9250: %d", ret);
        return;
    }

    int counter = 0;

    // Main loop
    while (1) {
        mpu9250_data_t sensor_data;
        ret = mpu9250_read_sensors(&sensor_data);
        
        if (counter == 0){
            if (ret == ESP_OK) {
                ESP_LOGI(TAG, "Accel: X=%.2f Y=%.2f Z=%.2f (g)",
                        sensor_data.accel_x, sensor_data.accel_y, sensor_data.accel_z);
                
                // ESP_LOGI(TAG, "Gyro: X=%.2f Y=%.2f Z=%.2f (deg/s)",
                        // sensor_data.gyro_x, sensor_data.gyro_y, sensor_data.gyro_z);
                
                // ESP_LOGI(TAG, "Mag: X=%.2f Y=%.2f Z=%.2f (uT)",
                        // sensor_data.mag_x, sensor_data.mag_y, sensor_data.mag_z);
                
                 if (xQueueSend(point_queue, &sensor_data, 0) == pdPASS) {
                    xEventGroupSetBits(event_group, POINT_QUEUE_BIT);
                }
            } else{
                ESP_LOGI(TAG, "Failed to obtained data from sensor");
            }
        }

        // counter += 1;
    }
}

void app_main(void)
{

    ESP_LOGI(TAG, "[APP] Startup..");
    ESP_LOGI(TAG, "[APP] Free memory: %ld bytes", esp_get_free_heap_size());
    ESP_LOGI(TAG, "[APP] IDF version: %s", esp_get_idf_version());

    ESP_LOGI(TAG, "ESP_WIFI_MODE_STA");

    init_nvs();
    wifi_init_sta();

    point_queue = xQueueCreate(100, sizeof(mpu9250_data_t));
    event_group = xEventGroupCreate();

    xTaskCreatePinnedToCore(
        read_sensor_task,
        "read_sensor_task",
        4096,
        NULL,
        5,
        NULL,
        0  // Pin to Core 1
    );

    xTaskCreatePinnedToCore(
        wifi_transmission_task,
        "wifi_transmission_task",
        4096,
        NULL,
        5,
        NULL,
        1  // Pin to Core 1
    );

}