// run.c - Fixed Inference code with Un-normalization
#include <xc.h>
#include <stdint.h>
#include "lcd.h"

#define _XTAL_FREQ 20000000

// Macro to convert float literals to Q8.8 fixed-point
#define F(x) ((int16_t)((x) * 256.0f))

// ============================================================================
// KEYPAD INPUT FUNCTIONS
// ============================================================================

const char keypad_layout[4][3] = {
    {'1', '2', '3'},
    {'4', '5', '6'},
    {'7', '8', '9'},
    {'*', '0', '#'}
};

char read_keypad(void) {
    for (uint8_t row = 0; row < 4; row++) {
        PORTB = 0xF0;
        PORTB &= ~(0x10 << row);
        __delay_us(10);
        
        for (uint8_t col = 0; col < 3; col++) {
            if (!(PORTB & (1 << col))) {
                __delay_ms(30);
                while (!(PORTB & (1 << col)));
                __delay_ms(30);
                return keypad_layout[row][col];
            }
        }
    }
    return 0;
}

int16_t read_decimal_from_keypad(void) {
    int16_t integer_part = 0;
    int16_t fractional_part = 0;
    int16_t fractional_divisor = 1;
    uint8_t reading_fraction = 0;
    char key;
    
    while (1) {
        key = read_keypad();
        if (!key) continue;
        if (key == '#') break;
        
        if (key == '*') {
            reading_fraction = 1;
            lcd_send_data('.');
            continue;
        }
        
        if (key >= '0' && key <= '9') {
            lcd_send_data(key);
            if (!reading_fraction) {
                integer_part = integer_part * 10 + (key - '0');
            } else {
                fractional_part = fractional_part * 10 + (key - '0');
                fractional_divisor *= 10;
            }
        }
    }
    
    return (int16_t)((integer_part << 8) + ((fractional_part << 8) / fractional_divisor));
}

// ============================================================================
// FIXED-POINT MATH & ACTIVATION
// ============================================================================

#define MULTIPLY_FIXED_POINT(a, b) ((int16_t)(((int32_t)(a) * (b)) >> 8))

const int16_t sigmoid_lookup_table[17] = {
    5, 8, 12, 19, 31, 47, 69, 97, 128, 159, 187, 209, 225, 237, 244, 249, 251
};

static inline int16_t calculate_sigmoid(int16_t z_value) {
    if (z_value <= -(4 << 8)) return sigmoid_lookup_table[0];
    if (z_value >= (4 << 8)) return sigmoid_lookup_table[16];
    
    uint8_t index = (uint8_t)(((z_value + (4 << 8)) >> 7));
    if (index > 16) index = 16;
    
    return sigmoid_lookup_table[index];
}

// ============================================================================
// TRAINED NETWORK PARAMETERS (Loaded from your LCD output)
// ============================================================================

#define NUM_HIDDEN_NEURONS 6

int16_t weights_input_to_hidden[NUM_HIDDEN_NEURONS] = { 
    F(0.19), F(-1.37), F(0.42), F(-1.54), F(0.66), F(-1.78) 
};

int16_t biases_hidden_layer[NUM_HIDDEN_NEURONS] = { 
    F(-1.15), F(-1.07), F(0), F(0.07), F(0.15), F(0.23) 
};

int16_t weights_hidden_to_output[NUM_HIDDEN_NEURONS] = { 
    F(0.11), F(0.17), F(0.23), F(0.29), F(0.35), F(0.41) 
};

int16_t bias_output_layer = F(0);

// ============================================================================
// FORWARD PASS & INFERENCE
// ============================================================================

int16_t neural_network_predict(int16_t input_x) {
    int32_t output_accumulator = bias_output_layer;
    
    for (uint8_t neuron = 0; neuron < NUM_HIDDEN_NEURONS; neuron++) {
        int16_t z = MULTIPLY_FIXED_POINT(weights_input_to_hidden[neuron], input_x) 
                  + biases_hidden_layer[neuron];
        
        int16_t h = calculate_sigmoid(z);
        
        output_accumulator += ((int32_t)weights_hidden_to_output[neuron] * h) >> 8;
    }
    
    return (int16_t)output_accumulator;
}

void run_network(void) {
    TRISB = 0x0F;
    OPTION_REGbits.nRBPU = 0; 
    PORTB = 0xFF;
    
    lcd_clear_screen();
    lcd_print_string("Inference Mode");
    __delay_ms(1000);

    while (1) {
        lcd_clear_screen();
        lcd_print_string("Enter x:");
        lcd_goto_position(2, 1);

        int16_t x_raw = read_decimal_from_keypad();

        // 1. Normalize x: x_norm = x_raw / 10
        int16_t x_norm = (int16_t)(((int32_t)x_raw) / 10);

        // 2. Predict (returns squashed value 0-1 in Q8.8)
        int16_t y_norm = neural_network_predict(x_norm);

        // 3. Un-normalize y: y_real = (y_norm * Range) + Min
        // Range = 350 (357 - 7), Min = 7
        int32_t y_temp = ((int32_t)y_norm * 350) >> 8;
        int16_t y_final = (int16_t)y_temp + 7;

        // 4. Display result
        lcd_clear_screen();
        lcd_print_string("y = ");
        lcd_print_unsigned_int((uint16_t)y_final);
        
        // Wait for any key to enter next value
        while(!read_keypad()); 
    }
}