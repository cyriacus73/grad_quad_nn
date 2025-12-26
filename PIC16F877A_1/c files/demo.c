// demo.c
// Interactive neural network demo - collects 10 training samples from user,
// trains the network, then allows testing predictions

#include <xc.h>
#include <stdint.h>
#include "lcd.h"

#define _XTAL_FREQ 20000000

// ============================================================================
// KEYPAD INPUT FUNCTIONS
// ============================================================================

const char keypad_map[4][3] = {
    {'1', '2', '3'},
    {'4', '5', '6'},
    {'7', '8', '9'},
    {'*', '0', '#'}
};

// Scan keypad and return pressed key (or 0 if none)
char read_key(void) {
    for (uint8_t row = 0; row < 4; row++) {
        PORTB = 0xF0;
        PORTB &= ~(0x10 << row);
        __delay_us(10);
        
        for (uint8_t col = 0; col < 3; col++) {
            if (!(PORTB & (1 << col))) {
                __delay_ms(30);  // Debounce
                while (!(PORTB & (1 << col)));  // Wait for release
                __delay_ms(30);
                return keypad_map[row][col];
            }
        }
    }
    return 0;
}

// Read decimal number from keypad (* = decimal, # = confirm)
int16_t read_number_q88(void) {
    int16_t integer_part = 0;
    int16_t fractional_part = 0;
    int16_t divisor = 1;
    uint8_t reading_fraction = 0;
    char key;
    
    while (1) {
        key = read_key();
        if (!key) continue;
        
        if (key == '#') break;  // Confirm
        
        if (key == '*') {  // Decimal point
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
                divisor *= 10;
            }
        }
    }
    
    return (integer_part << 8) + ((fractional_part << 8) / divisor);
}

// ============================================================================
// FIXED-POINT MATH
// ============================================================================

#define MULTIPLY_Q88(a, b) ((int16_t)(((int32_t)(a) * (b)) >> 8))

// ============================================================================
// SIGMOID ACTIVATION (Lookup Table)
// ============================================================================

// Sigmoid LUT for range [-4, 4] with step 0.5
const int16_t sigmoid_table[17] = {
    5, 10, 18, 32, 56, 96, 150, 198,
    256,  // sigmoid(0) = 0.5
    314, 362, 406, 440, 464, 480, 492, 500
};

// Calculate sigmoid(x) using lookup table
int16_t sigmoid_activation(int16_t x) {
    // Map x to index: x in [-4, 4] -> index in [0, 16]
    int16_t index = (x >> 7) + 8;  // (x / 128) + 8
    if (index < 0) index = 0;
    if (index > 16) index = 16;
    return sigmoid_table[index];
}

// Calculate sigmoid derivative: sig'(s) = s * (1 - s)
int16_t sigmoid_derivative(int16_t s) {
    return MULTIPLY_Q88(s, (256 - s));
}

// ============================================================================
// NEURAL NETWORK (3 hidden neurons, 10 training samples max)
// ============================================================================

#define NUM_HIDDEN 3
#define MAX_SAMPLES 10

// Network parameters
int16_t weights_input_hidden[NUM_HIDDEN] = { 64, 32, -32 };
int16_t biases_hidden[NUM_HIDDEN] = { 0, 0, 0 };
int16_t weights_hidden_output[NUM_HIDDEN] = { 64, 64, 64 };
int16_t bias_output = 0;

// Forward pass through network
int16_t network_forward(int16_t x, int16_t z[NUM_HIDDEN], int16_t h[NUM_HIDDEN]) {
    // Hidden layer
    for (uint8_t i = 0; i < NUM_HIDDEN; i++) {
        z[i] = MULTIPLY_Q88(weights_input_hidden[i], x) + biases_hidden[i];
        h[i] = sigmoid_activation(z[i]);
    }
    
    // Output layer
    int16_t output = bias_output;
    for (uint8_t i = 0; i < NUM_HIDDEN; i++) {
        output += MULTIPLY_Q88(weights_hidden_output[i], h[i]);
    }
    
    return output;
}

// Train network on provided data
void train_demo_network(int16_t* x_data, int16_t* y_data, uint8_t num_samples) {
    int16_t learning_rate = 16;  // Q8.8 format
    
    // Train for 60 epochs
    for (uint8_t epoch = 0; epoch < 60; epoch++) {
        // Process each training sample
        for (uint8_t sample = 0; sample < num_samples; sample++) {
            int16_t z[NUM_HIDDEN], h[NUM_HIDDEN];
            
            // Forward pass
            int16_t prediction = network_forward(x_data[sample], z, h);
            int16_t error = prediction - y_data[sample];
            
            // Update output layer
            bias_output -= MULTIPLY_Q88(learning_rate, error);
            for (uint8_t j = 0; j < NUM_HIDDEN; j++) {
                weights_hidden_output[j] -= MULTIPLY_Q88(learning_rate, MULTIPLY_Q88(error, h[j]));
            }
            
            // Update hidden layer
            for (uint8_t j = 0; j < NUM_HIDDEN; j++) {
                int16_t delta = MULTIPLY_Q88(
                    MULTIPLY_Q88(error, weights_hidden_output[j]),
                    sigmoid_derivative(h[j])
                );
                weights_input_hidden[j] -= MULTIPLY_Q88(learning_rate, MULTIPLY_Q88(delta, x_data[sample]));
                biases_hidden[j] -= MULTIPLY_Q88(learning_rate, delta);
            }
        }
    }
}

// ============================================================================
// DEMO APPLICATION
// ============================================================================

void run_demo(void) {
    // Configure keypad
    TRISB = 0x0F;  // Lower 4 bits input (columns)
    OPTION_REGbits.nRBPU = 0;  // Enable pull-ups
    PORTB = 0xFF;
    
    lcd_initialize();
    
    int16_t x_dataset[MAX_SAMPLES];
    int16_t y_dataset[MAX_SAMPLES];
    
    // Collect training data from user
    for (uint8_t i = 0; i < MAX_SAMPLES; i++) {
        // Get X value
        lcd_clear_screen();
        lcd_print_string("X:");
        lcd_goto_position(2, 1);
        int16_t x = read_number_q88();
        
        // Get Y value
        lcd_clear_screen();
        lcd_print_string("Y:");
        lcd_goto_position(2, 1);
        int16_t y = read_number_q88();
        
        // Store normalized data
        x_dataset[i] = x / 10;  // Normalize x
        y_dataset[i] = y;
        
        // Confirm saved
        lcd_clear_screen();
        lcd_print_unsigned_int(i + 1);
        lcd_print_string("/10 saved");
        __delay_ms(800);
    }
    
    // Train the network
    lcd_clear_screen();
    lcd_print_string("Training...");
    train_demo_network(x_dataset, y_dataset, MAX_SAMPLES);
    __delay_ms(1500);
    
    // Testing loop
    while (1) {
        // Prompt for test input
        lcd_clear_screen();
        lcd_print_string("Test X:");
        lcd_goto_position(2, 1);
        int16_t test_x = read_number_q88();
        
        // Get prediction
        int16_t z[NUM_HIDDEN], h[NUM_HIDDEN];
        int16_t predicted_y = network_forward(test_x / 10, z, h);
        
        // Display result
        lcd_clear_screen();
        lcd_print_string("Y=");
        lcd_print_fixed_point(predicted_y);
        __delay_ms(3000);
    }
}