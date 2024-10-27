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
    int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);

    struct sockaddr_in dest_addr;
    dest_addr.sin_addr.s_addr = inet_addr(CLIENT_IP);
    dest_addr.sin_family = AF_INET;
    dest_addr.sin_port = htons(PORT);

    // Set socket buffer size
    int sendbuff = 64000;  // 16KB buffer
    setsockopt(sock, SOL_SOCKET, SO_SNDBUF, &sendbuff, sizeof(sendbuff));

    while (1) {
        if (sock < 0) {
            ESP_LOGE(TAG, "Unable to create socket: errno %d", errno);
            if (sock != -1) {
                shutdown(sock, 0);
                close(sock);
            }
            sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
            continue;
        }

        EventBits_t bits = xEventGroupWaitBits(
            event_group,
            POINT_QUEUE_BIT,
            pdTRUE,
            pdFALSE,
            pdMS_TO_TICKS(1)
        );

        if (bits & POINT_QUEUE_BIT) {
            // Clear the queue by reading until empty, keeping only the last value
            mpu9250_data_t latest_data;
            while (xQueueReceive(point_queue, &latest_data, 0) == pdPASS) {
                data = latest_data;  // Keep overwriting with newer values
            }

            // Send the most recent value
            int err = sendto(sock, &data, sizeof(mpu9250_data_t), 0, 
                           (struct sockaddr *)&dest_addr, sizeof(dest_addr));
            
            if (err < 0) {
                ESP_LOGE(TAG, "Error during sending: errno %d", errno);
                vTaskDelay(pdMS_TO_TICKS(5));  // Small backoff on error
            }
        }

    }

    shutdown(sock, 0);
    close(sock);
}

void read_sensor_task(void *pvParameters) {
    mpu9250_config_t mpu_config = {
        .i2c_port = I2C_NUM_0,
        .sda_pin = GPIO_NUM_26,
        .scl_pin = GPIO_NUM_25,
        .clk_speed = 400000
    };

    esp_err_t ret = mpu9250_init(&mpu_config);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize MPU9250: %d", ret);
        return;
    }

    // TickType_t last_wake_time = xTaskGetTickCount();
    // const TickType_t frequency = 1;

    while (1) {
        mpu9250_data_t sensor_data;
        ret = mpu9250_read_sensors(&sensor_data);
        
        if (ret == ESP_OK) {
            // Overwrite old data in queue - if queue is full, overwrite the oldest value
            xQueueOverwrite(point_queue, &sensor_data);
            xEventGroupSetBits(event_group, POINT_QUEUE_BIT);
        } else {
            ESP_LOGI(TAG, "Failed to obtain data from sensor");
        }
        
        // Maintain precise timing for sensor reads
        vTaskDelay(pdMS_TO_TICKS(1));
        // vTaskDelayUntil(&last_wake_time, frequency);
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

    point_queue = xQueueCreate(1, sizeof(mpu9250_data_t));
    event_group = xEventGroupCreate();

    xTaskCreatePinnedToCore(
        read_sensor_task,
        "read_sensor_task",
        4096,
        NULL,
        5,
        NULL,
        0  // Pin to Core 0
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