#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "mpu9250.h"
#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_netif.h"
#include "esp_mac.h"
#include "esp_event.h"
#include "esp_timer.h"
#include "lwip/sockets.h"
#include "nvs_flash.h"
#include "nvs.h"
#include <inttypes.h>


#define ESP_WIFI_SSID      CONFIG_ESP_WIFI_SSID
#define ESP_WIFI_PASS      CONFIG_ESP_WIFI_PASSWORD
#define ESP_MAXIMUM_RETRY  CONFIG_ESP_MAXIMUM_RETRY

#define PORT 8035
#define BUFFER_LENGTH 1500

#define POINT_QUEUE_BIT (1 << 1)

#define CLIENT_IP  "192.168.4.2" // CONFIG_CLIENT_IP , IP of client you'd like to connect to
#define HOST_IP "192.168.4.1"  // ESP32 AP IP

typedef struct __attribute__ ((packed)) {
   mpu9250_data_t mpu;
   int64_t timestamp;
} pose_packet;

#define PACKET_SIZE sizeof(pose_packet) // Bytes


QueueHandle_t point_queue;

static const char *TAG = "MAIN";

void init_nvs() {
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(err);
    printf("Did err? => (%s)\n", esp_err_to_name(err));
}

void wifi_event_handler(void* arg, esp_event_base_t event_base,
                       int32_t event_id, void* event_data)
{
    if (event_base == WIFI_EVENT) {
        switch (event_id) {
            case WIFI_EVENT_AP_STACONNECTED: {
                wifi_event_ap_staconnected_t* event = (wifi_event_ap_staconnected_t*) event_data;
                ESP_LOGI(TAG, "Station "MACSTR" joined, AID=%d",
                         MAC2STR(event->mac), event->aid);
                break;
            }
            case WIFI_EVENT_AP_STADISCONNECTED: {
                wifi_event_ap_stadisconnected_t* event = (wifi_event_ap_stadisconnected_t*) event_data;
                ESP_LOGI(TAG, "Station "MACSTR" left, AID=%d",
                         MAC2STR(event->mac), event->aid);
                break;
            }
        }
    }
}

void wifi_init_ap(void)
{

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    esp_netif_t *ap_netif = esp_netif_create_default_wifi_ap();
    // Configure the AP's IP address and DHCP server
    esp_netif_ip_info_t ip_info;
    IP4_ADDR(&ip_info.ip, 192, 168, 4, 1);      // ESP32 AP IP address
    IP4_ADDR(&ip_info.gw, 192, 168, 4, 1);      // Gateway (same as IP)
    IP4_ADDR(&ip_info.netmask, 255, 255, 255, 0);

    esp_netif_dhcps_stop(ap_netif);
    esp_netif_set_ip_info(ap_netif, &ip_info);
    esp_netif_dhcps_start(ap_netif);

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

     // Register event handler
    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, 
                                             ESP_EVENT_ANY_ID, 
                                             &wifi_event_handler, 
                                             NULL));

    wifi_config_t wifi_config = {
       .ap = {
        .ssid = "Drone_Studio",
        .ssid_len = strlen("Drone_Studio"),
        .channel = 11,        // Try 5GHz band if your hardware supports it
        .password = "your_password",
        .max_connection = 1,
        .authmode = WIFI_AUTH_WPA2_PSK
        // .beacon_interval = 100  // Minimize beacon interval
        },
    };

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_AP));
    ESP_ERROR_CHECK(esp_wifi_set_config(ESP_IF_WIFI_AP, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_set_bandwidth(ESP_IF_WIFI_AP, WIFI_BW_HT40));
    ESP_ERROR_CHECK(esp_wifi_set_protocol(ESP_IF_WIFI_AP, WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N));
    ESP_ERROR_CHECK(esp_wifi_config_80211_tx_rate(ESP_IF_WIFI_AP, WIFI_PHY_RATE_MCS7_SGI));
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "wifi_init_ap finished. SSID:%s password:%s channel:%d",
             wifi_config.ap.ssid, wifi_config.ap.password, wifi_config.ap.channel);

}


void wifi_transmission_task(void *pvParameters) {
    pose_packet data;
    int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);

    // Set non-blocking
    int flags = fcntl(sock, F_GETFL, 0);
    fcntl(sock, F_SETFL, flags | O_NONBLOCK);

    // Statistics variables
    uint32_t packets_sent = 0;
    uint32_t last_print = 0;
    uint32_t send_errors = 0;

    
    // Increase buffer size
    int sendbuff = 262144;
    setsockopt(sock, SOL_SOCKET, SO_SNDBUF, &sendbuff, sizeof(sendbuff));

    struct sockaddr_in dest_addr;
    dest_addr.sin_addr.s_addr = inet_addr(CLIENT_IP);
    dest_addr.sin_family = AF_INET;
    dest_addr.sin_port = htons(PORT);


    while (1) {
        uint32_t now = xTaskGetTickCount();
        int64_t tick = esp_timer_get_time();

        if (sock < 0) {
            ESP_LOGE(TAG, "Unable to create socket: errno %d", errno);
            sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }

        // Clear the queue by reading until empty, keeping only the last value
        if (xQueueReceive(point_queue, &data, 0) == pdTRUE) {
            data.timestamp = tick;
            int err = sendto(sock, &data, PACKET_SIZE, 0, 
                           (struct sockaddr *)&dest_addr, sizeof(dest_addr));
            
            if (err < 0) {
                // ESP_LOGE(TAG, "Error sending data: errno %d", errno);
                vTaskDelay(1);
            }
            else {
                packets_sent++;
            }
        }

         // Print stats every second
        if ((now - last_print) >= pdMS_TO_TICKS(1000)) {
            ESP_LOGI(TAG, "Tick: %"PRId64", Packets/sec: %lu, Errors: %lu", 
                    tick, packets_sent, send_errors);
            last_print = now;
            packets_sent = 0;
        }
    }

    if (sock != -1) {
        shutdown(sock, 0);
        close(sock);
    }
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

    vTaskPrioritySet(NULL, configMAX_PRIORITIES - 1);

    while (1) {
        mpu9250_data_t sensor_data;
        ret = mpu9250_read_sensors(&sensor_data);
        
        if (ret == ESP_OK) {
            xQueueOverwrite(point_queue, &sensor_data);
        } else {
            ESP_LOGI(TAG, "Failed to obtain data from sensor");
        }
        
        vTaskDelay(pdMS_TO_TICKS(1));
    }
}

void app_main(void)
{

    ESP_LOGI(TAG, "[APP] Startup..");
    ESP_LOGI(TAG, "[APP] Free memory: %ld bytes", esp_get_free_heap_size());
    ESP_LOGI(TAG, "[APP] IDF version: %s", esp_get_idf_version());

    ESP_LOGI(TAG, "ESP_WIFI_MODE_STA");

    init_nvs();
    wifi_init_ap();

    point_queue = xQueueCreate(1, sizeof(mpu9250_data_t));

    xTaskCreatePinnedToCore(
        read_sensor_task,
        "read_sensor_task",
        4096,
        NULL,
        configMAX_PRIORITIES - 2,
        NULL,
        0  // Pin to Core 0
    );

    xTaskCreatePinnedToCore(
        wifi_transmission_task,
        "wifi_transmission_task",
        4096,
        NULL,
        configMAX_PRIORITIES - 1,
        NULL,
        1  // Pin to Core 1
    );

    // while (1) {
    //     print_sta_info();
    //     vTaskDelay(pdMS_TO_TICKS(5000));
    // }

}