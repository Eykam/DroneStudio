#include "mpu9250.h"
#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "lwip/err.h"
#include "lwip/sockets.h"
#include "lwip/sys.h"
#include <lwip/netdb.h>

#define ESP_WIFI_SSID      CONFIG_ESP_WIFI_SSID
#define ESP_WIFI_PASS      CONFIG_ESP_WIFI_PASSWORD
#define ESP_MAXIMUM_RETRY  CONFIG_ESP_MAXIMUM_RETRY

#define PORT 8035
#define BUFFER_LENGTH 1500

/* FreeRTOS event group to signal when we are connected*/
static EventGroupHandle_t s_wifi_event_group;
static EventGroupHandle_t event_group;

#define STATUS_QUEUE_BIT (1 << 0)
#define POINT_QUEUE_BIT (1 << 1)

#define CLIENT_IP <CLIENT_IP_HERE> // IP of client you'd like to connect to
#define HOST_IP 

#define WIFI_CONNECTED_BIT BIT0
#define WIFI_FAIL_BIT      BIT1

static const char *TAG = "MAIN";

QueueHandle_t point_queue;

typedef struct {
    uint16_t horizontal_steps;  // 8 bits (8 unused)
    uint16_t vertical_steps;    // 6 bits (10 bits unused)
    uint32_t distance_mm;      // 18 bits (14 bits unused)
} ScannerData;

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
            /* Setting a password implies station will connect to all security modes including WEP/WPA.
             * However these modes are deprecated and not advisable to be used. Incase your Access point
             * doesn't support WPA2, these mode can be enabled by commenting below line */
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

    ESP_LOGI(tag, "wifi_init_sta finished.");

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
        ESP_LOGI(tag, "connected to ap SSID:%s",
                 ESP_WIFI_SSID);
    } else if (bits & WIFI_FAIL_BIT) {
        ESP_LOGI(tag, "Failed to connect to SSID:%s",
                 ESP_WIFI_SSID);
    } else {
        ESP_LOGE(tag, "UNEXPECTED EVENT");
    }

    ESP_ERROR_CHECK(esp_event_handler_unregister(IP_EVENT, IP_EVENT_STA_GOT_IP, &event_handler));
    ESP_ERROR_CHECK(esp_event_handler_unregister(WIFI_EVENT, ESP_EVENT_ANY_ID, &event_handler));
    vEventGroupDelete(s_wifi_event_group);
}

void wifi_transmission_task(void *pvParameters) {
    ScannerData data;
    uint8_t received_status;
    
    // Create a UDP socket
    int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);

    struct sockaddr_in dest_addr;
    dest_addr.sin_addr.s_addr = inet_addr(CLIENT_IP);
    dest_addr.sin_family = AF_INET;
    dest_addr.sin_port = htons(8035);

    while (1) {
        if (sock < 0) {
            ESP_LOGE(tag, "Unable to create socket: errno %d", errno);
            sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
            continue;
        }

        // Check both queues without blocking
        EventBits_t bits = xEventGroupWaitBits(
            event_group,
            STATUS_QUEUE_BIT | POINT_QUEUE_BIT,
            pdTRUE,  // Clear bits before returning
            pdFALSE,  // Don't wait for all bits
            0  // Don't block
        );

        if (bits & STATUS_QUEUE_BIT) {
            if (xQueueReceive(status_queue, &received_status, 0) == pdPASS) {
                uint8_t packet[1];
                packet[0] = received_status;
                Status = received_status;
            
                ESP_LOGI(tag, "Sending status to backend => %d", Status);
                int err = sendto(sock, packet, sizeof(packet), 0, (struct sockaddr *)&dest_addr, sizeof(dest_addr));

                if (err < 0) {
                    ESP_LOGE(tag, "Error occurred during sending done command: errno %d", errno);
                }
            }
        }

        if (bits & POINT_QUEUE_BIT) {
            if (data.vertical_steps < VERTICAL_MAX_STEPS && xQueueReceive(point_queue, &data, 0) == pdPASS) {
                uint16_t packet[PACKET_SIZE];
                // Pack data into the packet
                packet[0] = data.horizontal_steps;
                packet[1] = data.vertical_steps;  // Only use 6 bits
                packet[2] = (data.distance_mm >> 8) & 0xFF;  // High byte
                packet[3] = data.distance_mm & 0xFF;  // Low byte

                int err = sendto(sock, packet, sizeof(packet), 0, (struct sockaddr *)&dest_addr, sizeof(dest_addr));
            
                if (err < 0) {
                    ESP_LOGE(tag, "Error occurred during sending: errno %d", errno);
                }
            }
        }

        // If neither queue had data, add a small delay to prevent tight-looping
        if ((bits & (STATUS_QUEUE_BIT | POINT_QUEUE_BIT)) == 0) {
            vTaskDelay(pdMS_TO_TICKS(10));
        }
    }

    shutdown(sock, 0);
    close(sock);
}

void app_main(void)
{

    ESP_LOGI(tag, "[APP] Startup..");
    ESP_LOGI(tag, "[APP] Free memory: %ld bytes", esp_get_free_heap_size());
    ESP_LOGI(tag, "[APP] IDF version: %s", esp_get_idf_version());

    ESP_LOGI(tag, "ESP_WIFI_MODE_STA");
    wifi_init_sta();

    status_queue = xQueueCreate(100, sizeof(uint8_t));
    event_group = xEventGroupCreate();


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


    xTaskCreatePinnedToCore(
        wifi_transmission_task,
        "wifi_transmission_task",
        4096,
        NULL,
        5,
        NULL,
        1  // Pin to Core 1
    );

    int counter = 0;
    // Main loop
    while (1) {
        mpu9250_data_t sensor_data;
        ret = mpu9250_read_sensors(&sensor_data);
        
        if (counter % 150 == 0){
            if (ret == ESP_OK) {
                // ESP_LOGI(TAG, "Accel: X=%.2f Y=%.2f Z=%.2f (g)",
                //         sensor_data.accel_x, sensor_data.accel_y, sensor_data.accel_z);
                
                // ESP_LOGI(TAG, "Gyro: X=%.2f Y=%.2f Z=%.2f (deg/s)",
                //         sensor_data.gyro_x, sensor_data.gyro_y, sensor_data.gyro_z);
                
                ESP_LOGI(TAG, "Mag: X=%.2f Y=%.2f Z=%.2f (uT)",
                        sensor_data.mag_x, sensor_data.mag_y, sensor_data.mag_z);
            } else{
                ESP_LOGI(TAG, "Failed to obtained data from sensor");
            }
        }

        counter += 1;
    }
}