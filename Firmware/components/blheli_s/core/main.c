#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/rmt.h"
#include "driver/gpio.h"
#include "driver/adc.h"

// Configuration Definitions
#define RMT_TX_CHANNEL      RMT_CHANNEL_0        // RMT channel to use
#define RMT_TX_GPIO_NUM     GPIO_NUM_32          // GPIO32 for DShot600 signal
#define RMT_CLK_DIV         8                    // Clock divider (80MHz / 8 = 10MHz)
#define DSHOT_BIT_RATE      600000               // DShot600 bit rate (600 kbps)
#define DSHOT_TICKS_PER_BIT 16                   // Total ticks per DShot bit (1.6us per bit at 0.1us/tick)
#define DSHOT_RESET_TICKS   5000                 // Reset pulse duration (500us = 5000 ticks)

// DShot Command Definitions
#define DSHOT_ARMING_CMD    0                    // BLHELI_S arming command
#define DSHOT_MIN_THROTTLE  48                   // Minimum throttle value for DShot
#define DSHOT_MAX_THROTTLE  2047                 // Maximum throttle value for DShot (actual max is 2047)
#define DSHOT_FRAME_DELAY   20                   // Delay between frames in milliseconds


// Button GPIO Definitions
#define BUTTON_ARM_GPIO             GPIO_NUM_33  // GPIO for min throttle button
#define BUTTON_EVENT_DISARM_GPIO    GPIO_NUM_25  // GPIO for max throttle button
#define BUTTON_25_PERCENT_GPIO      GPIO_NUM_26  // GPIO for 25% throttle button

#define POTENTIOMETER_GPIO  GPIO_NUM_34 // GPIO for potentiometer
#define POTENTIOMETER_CHANNEL  ADC1_CHANNEL_6

// Function Prototypes
uint8_t dshot_checksum(uint16_t command);
void send_dshot_command(rmt_channel_t channel, uint16_t throttle);
void init_rmt();
void init_buttons();
void button_task(void *arg);
void throttle_task(void *arg);

typedef enum {
    BUTTON_EVENT_ARM,
    BUTTON_EVENT_DISARM,
    BUTTON_EVENT_25_PERCENT
} button_event_t;

typedef enum {
    ESC_STATE_DISARMED,
    ESC_STATE_ARMING,
    ESC_STATE_ARMED
} esc_state_t;

QueueHandle_t button_event_queue;


volatile uint16_t throttle_value = DSHOT_MIN_THROTTLE;
volatile bool throttle_on = false;
volatile esc_state_t esc_state = ESC_STATE_DISARMED;

uint8_t dshot_checksum(uint16_t command) {
    return (command ^ (command >> 4) ^ (command >> 8)) & 0x0F;
}

void send_dshot_command(rmt_channel_t channel, uint16_t throttle) {
    // Ensure throttle is within valid range
    if(throttle > DSHOT_MAX_THROTTLE) throttle = DSHOT_MAX_THROTTLE;
    

    // Construct the 16-bit DShot command
    uint16_t command = throttle << 1; // Left shift to make room for telemetry bit
    command |= 0; // Telemetry bit set to 0

    // Calculate checksum and append
    uint8_t checksum = dshot_checksum(command);
    command = (command << 4) | checksum; // Append checksum

    // Prepare RMT items for 16 bits
    rmt_item32_t items[17]; // 16 bits + reset pulse
    for(int i = 0; i < 16; i++) {
        uint8_t bit = (command >> (15 - i)) & 0x1;
        if(bit) {
            // Logical '1': High for ~1.2us (12 ticks), Low for ~0.4us (4 ticks)
            items[i].level0 = 1;
            items[i].duration0 = 12; // 12 ticks * 0.1us = 1.2us
            items[i].level1 = 0;
            items[i].duration1 = 4;  // 4 ticks * 0.1us = 0.4us
        }
        else {
            // Logical '0': High for ~0.8us (8 ticks), Low for ~0.8us (8 ticks)
            items[i].level0 = 1;
            items[i].duration0 = 6;  // 8 ticks * 0.1us = 0.8us ??? Figure out why this works
            items[i].level1 = 0;
            items[i].duration1 = 10;  // 8 ticks * 0.1us = 0.8us ??? Figure out why this works
        }
    }

    // Reset pulse (ensure line stays low for at least 300us)
    items[16].level0 = 0;
    items[16].duration0 = DSHOT_RESET_TICKS;
    items[16].level1 = 0;
    items[16].duration1 = 0;

    // Send the DShot command
    esp_err_t ret = rmt_write_items(channel, items, 17, true);
    if(ret != ESP_OK) {
        printf("Error sending DShot command: %d\n", ret);
    }

    // Wait for transmission to complete
    rmt_wait_tx_done(channel, portMAX_DELAY);
}

// GPIO ISR handler
static void IRAM_ATTR gpio_isr_handler(void* arg) {
    uint32_t gpio_num = (uint32_t) arg;
    button_event_t event;

    if(gpio_num == BUTTON_ARM_GPIO) {
        event = BUTTON_EVENT_ARM;
    } else if(gpio_num == BUTTON_EVENT_DISARM_GPIO) {
        event = BUTTON_EVENT_DISARM;
    } else if(gpio_num == BUTTON_25_PERCENT_GPIO) {
        event = BUTTON_EVENT_25_PERCENT;
    } else {
        return;
    }

    xQueueSendFromISR(button_event_queue, &event, NULL);
}

void init_buttons() {
    gpio_config_t io_conf = {};
    io_conf.intr_type = GPIO_INTR_NEGEDGE;
    io_conf.mode = GPIO_MODE_INPUT;
    io_conf.pin_bit_mask = ((1ULL << BUTTON_ARM_GPIO) |
                           (1ULL << BUTTON_EVENT_DISARM_GPIO) |
                           (1ULL << BUTTON_25_PERCENT_GPIO));
    io_conf.pull_up_en = 1;
    io_conf.pull_down_en = 0;
    gpio_config(&io_conf);

    gpio_install_isr_service(ESP_INTR_FLAG_LEVEL3);

    gpio_isr_handler_add(BUTTON_ARM_GPIO, gpio_isr_handler, (void*) BUTTON_ARM_GPIO);
    gpio_isr_handler_add(BUTTON_EVENT_DISARM_GPIO, gpio_isr_handler, (void*) BUTTON_EVENT_DISARM_GPIO);
    gpio_isr_handler_add(BUTTON_25_PERCENT_GPIO, gpio_isr_handler, (void*) BUTTON_25_PERCENT_GPIO);
}

void arming_task(void *arg) {
    while(1) {
        if(esc_state == ESC_STATE_ARMING) {
            printf("Arming...\n");
            
            // Send arming command multiple times
            for (int i = 0; i < 1000; i++) {
                send_dshot_command(RMT_TX_CHANNEL, DSHOT_ARMING_CMD);
                vTaskDelay(pdMS_TO_TICKS(1));
            }
            
            if(esc_state == ESC_STATE_ARMING) { // Check if we weren't interrupted
                esc_state = ESC_STATE_ARMED;
                printf("Arming sequence complete. ESC is armed!\n");
            }

        }

        // while (on) {
        while (!throttle_on && esc_state == ESC_STATE_ARMED) {
            send_dshot_command(RMT_TX_CHANNEL, DSHOT_MIN_THROTTLE);
            vTaskDelay(pdMS_TO_TICKS(1)); // 1000Hz timing
        }

        vTaskDelay(pdMS_TO_TICKS(100)); // Check state every 100ms
    }
}

void button_task(void *arg) {
    button_event_t event;
    while(1) {
        if(xQueueReceive(button_event_queue, &event, portMAX_DELAY)) {
            switch(event) {
                case BUTTON_EVENT_ARM:
                    switch (esc_state) {
                        case ESC_STATE_ARMED:
                            printf("Disarming ESC...\n");
                            esc_state = ESC_STATE_DISARMED;
                            throttle_on = false;
                            break;
                        case ESC_STATE_DISARMED:
                            printf("Starting arming sequence...\n");
                            esc_state = ESC_STATE_ARMING;
                            break;
                        case ESC_STATE_ARMING:
                            break;
                        default:
                            printf("Unexpected condition in Arming event!\n");
                            break;
                    }

                    break;
                    
                case BUTTON_EVENT_DISARM:
                    printf("Stopping ESC...\n");
                    esc_state = ESC_STATE_DISARMED;
                    throttle_on = false;
                    break;
                    
                case BUTTON_EVENT_25_PERCENT:
                    switch (esc_state) {
                        case ESC_STATE_ARMING:
                            printf("Currently arming, please wait...\n");
                            break;  
                        case ESC_STATE_ARMED:
                            // throttle_value = <set throttle here>;                            
                            if (throttle_on){
                                printf("Pausing throttle...\n");
                                throttle_on = false;
                            }
                            else{
                                printf("Starting throttle at 25%%...\n");
                                throttle_on = true;
                            }
                            break;
                        case ESC_STATE_DISARMED:
                            printf("Cannot send throttle - ESC not armed!\n");
                            break;
                        default:
                            printf("Unexpected condition in throttle event!\n");
                            break;
                    }

                    break;
        
                default:
                    break;
            }
        }
    }
}

void throttle_task(void *arg) {
    float v_max = .950;
    float d_max = 4095;
    float clamped_throttle = (DSHOT_MAX_THROTTLE * 0.5) - DSHOT_MIN_THROTTLE;
    float max_input_throttle = .80;
    int count = 0;

    while(1) {
        if (count % 100 == 0){
            // uint16_t throttle_speed = DSHOT_MIN_THROTTLE + ((DSHOT_MAX_THROTTLE - DSHOT_MIN_THROTTLE) * 0.02);
            int adc_value = adc1_get_raw(POTENTIOMETER_CHANNEL);
            float voltage = v_max * (adc_value / d_max); // 0-0.8845V range
            float normalized_input = (voltage / max_input_throttle);

            if (normalized_input > 1.0) normalized_input = 1.0;

            throttle_value = DSHOT_MIN_THROTTLE + (uint16_t)(normalized_input * (clamped_throttle));
            printf("Throttle value: %.2f%%\n", normalized_input * 100);
        }

        if(throttle_on && esc_state == ESC_STATE_ARMED) {
            send_dshot_command(RMT_TX_CHANNEL, throttle_value);
            vTaskDelay(pdMS_TO_TICKS(1)); // 50Hz timing
        } else {
            vTaskDelay(pdMS_TO_TICKS(100));
        }

        count++;
    }
}

void init_rmt() {
    // Configure RMT for transmission
    rmt_config_t config = {
        .rmt_mode = RMT_MODE_TX,
        .channel = RMT_TX_CHANNEL,
        .gpio_num = RMT_TX_GPIO_NUM,
        .clk_div = RMT_CLK_DIV, // 80MHz / 8 = 10MHz (0.1us per tick)
        .mem_block_num = 1,
        .tx_config = {
            .loop_en = false,                // No loop
            .carrier_en = false,             // No carrier
            .idle_level = RMT_IDLE_LEVEL_LOW,
            .idle_output_en = true,          // Output idle level
        },
    };

    // Initialize RMT
    rmt_config(&config);
    rmt_driver_install(config.channel, 0, 0);
}

void app_main(void) {
    printf("Initializing DShot600 on ESP32 (GPIO32)...\n");

    // gpio_pulldown_en(RMT_TX_GPIO_NUM);

    init_rmt();
    init_buttons();

    // Initialize ADC for potentiometer
    adc1_config_width(ADC_WIDTH_BIT_12);
    adc1_config_channel_atten(POTENTIOMETER_CHANNEL, ADC_ATTEN_DB_0);

    button_event_queue = xQueueCreate(10, sizeof(button_event_t));


    xTaskCreate(button_task, "button_task", 2048, NULL, 10, NULL);
    xTaskCreate(throttle_task, "throttle_task", 2048, NULL, 10, NULL);
    xTaskCreate(arming_task, "arming_task", 2048, NULL, 10, NULL);

    // Allow ESC to initialize
    vTaskDelay(pdMS_TO_TICKS(3000));
    printf("Initialization complete. Press ARM button to begin arming sequence...\n");

    vTaskDelete(NULL);
}