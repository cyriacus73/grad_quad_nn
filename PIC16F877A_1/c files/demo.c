// demo.c
// Interactive neural network demo - collects 10 user-provided training samples,
// trains a small network, then lets the user test predictions
#include <xc.h>
#include <stdint.h>
#include "lcd.h"
#include "nn.h"   // for QMUL macro
#define _XTAL_FREQ 20000000

// ============================================================================
// KEYPAD INPUT
// ============================================================================
const char keypad_map[4][3] = {
    {'1', '2', '3'},
    {'4', '5', '6'},
    {'7', '8', '9'},
    {'*', '0', '#'}
};

char read_key(void) {
    for (uint8_t row = 0; row < 4; row++) {
        PORTB = 0xF0;
        PORTB &= ~(0x10 << row);
        __delay_us(10);

        for (uint8_t col = 0; col < 3; col++) {
            if (!(PORTB & (1 << col))) {
                __delay_ms(30);               // Debounce
                while (!(PORTB & (1 << col))); // Wait release
                __delay_ms(30);
                return keypad_map[row][col];
            }
        }
    }
    return 0;
}

// Read Q8.8 fixed-point number from keypad
// Supports: digits, * = decimal point, # = confirm, - (via special handling)
int16_t read_number_q88(void) {
    int32_t value = 0;
    uint8_t decimal_places = 0;
    uint8_t negative = 0;
    uint8_t has_decimal = 0;
    char key;

    lcd_print_string("Enter num (#=OK):");

    while (1) {
        key = read_key();
        if (!key) continue;

        if (key == '#') break;              // Confirm

        if (key == '*') {                   // Decimal point
            if (!has_decimal) {
                has_decimal = 1;
                lcd_send_data('.');
            }
            continue;
        }

        if (key == '-') {                   // Negative sign (only at start)
            if (value == 0 && !negative) {
                negative = 1;
                lcd_send_data('-');
            }
            continue;
        }

        if (key >= '0' && key <= '9') {
            lcd_send_data(key);
            if (has_decimal) {
                decimal_places++;
                value = value * 10 + (key - '0');
            } else {
                value = value * 10 + (key - '0');
            }
        }
    }

    // Scale to Q8.8
    if (decimal_places > 0) {
        uint32_t divisor = 1;
        for (uint8_t i = 0; i < decimal_places; i++) divisor *= 10;
        value = (value << 8) / divisor;
    } else {
        value <<= 8;
    }

    if (negative) value = -value;

    return (int16_t)value;
}

// ============================================================================
// SIGMOID (small LUT for demo)
// ============================================================================
const int16_t sigmoid_table[17] = {
    5, 10, 18, 32, 56, 96, 150, 198, 256, 314, 362, 406, 440, 464, 480, 492, 500
};

int16_t sigmoid_activation(int16_t x) {
    int16_t index = (x >> 7) + 8;  // -4..4 ? 0..16
    if (index < 0) index = 0;
    if (index > 16) index = 16;
    return sigmoid_table[index];
}

int16_t sigmoid_derivative(int16_t s) {
    return QMUL(s, (256 - s));
}

// ============================================================================
// DEMO NETWORK (3 hidden neurons)
// ============================================================================
#define NUM_HIDDEN 3
#define MAX_SAMPLES 10

int16_t w_ih[NUM_HIDDEN] = {64, 32, -32};
int16_t b_h[NUM_HIDDEN]  = {0, 0, 0};
int16_t w_ho[NUM_HIDDEN] = {64, 64, 64};
int16_t b_o = 0;

int16_t network_forward(int16_t x, int16_t z[NUM_HIDDEN], int16_t h[NUM_HIDDEN]) {
    for (uint8_t i = 0; i < NUM_HIDDEN; i++) {
        z[i] = QMUL(w_ih[i], x) + b_h[i];
        h[i] = sigmoid_activation(z[i]);
    }

    int16_t out = b_o;
    for (uint8_t i = 0; i < NUM_HIDDEN; i++) {
        out += QMUL(w_ho[i], h[i]);
    }
    return out;
}

void train_demo_network(int16_t* x_data, int16_t* y_data, uint8_t n_samples) {
    int16_t lr = 16;  // Q8.8 learning rate

    for (uint8_t epoch = 0; epoch < 80; epoch++) {  // more epochs for better fit
        for (uint8_t s = 0; s < n_samples; s++) {
            int16_t z[NUM_HIDDEN], h[NUM_HIDDEN];
            int16_t pred = network_forward(x_data[s], z, h);
            int16_t error = pred - y_data[s];

            // Output layer
            b_o -= QMUL(lr, error);
            for (uint8_t j = 0; j < NUM_HIDDEN; j++) {
                w_ho[j] -= QMUL(lr, QMUL(error, h[j]));
            }

            // Hidden layer
            for (uint8_t j = 0; j < NUM_HIDDEN; j++) {
                int16_t delta = QMUL(QMUL(error, w_ho[j]), sigmoid_derivative(h[j]));
                w_ih[j] -= QMUL(lr, QMUL(delta, x_data[s]));
                b_h[j] -= QMUL(lr, delta);
            }
        }

        if (epoch % 20 == 0) {
            lcd_goto_position(2, 1);
            lcd_print_string("Epoch:");
            lcd_print_unsigned_int(epoch);
        }
    }
}

// ============================================================================
// MAIN DEMO FLOW
// ============================================================================
void run_demo(void) {
    // Keypad setup (same as main.c)
    TRISB = 0x0F;
    OPTION_REGbits.nRBPU = 0;
    PORTB = 0xFF;

    lcd_clear_screen();
    lcd_print_string("Demo Mode");
    __delay_ms(1500);

    int16_t x_data[MAX_SAMPLES];
    int16_t y_data[MAX_SAMPLES];

    // Collect 10 samples
    for (uint8_t i = 0; i < MAX_SAMPLES; i++) {
        lcd_clear_screen();
        lcd_print_string("Sample ");
        lcd_print_unsigned_int(i + 1);
        lcd_print_string("/10");

        lcd_goto_position(2, 1);
        lcd_print_string("X = ");
        x_data[i] = read_number_q88() / 10;  // normalize input range

        lcd_clear_screen();
        lcd_print_string("Sample ");
        lcd_print_unsigned_int(i + 1);
        lcd_print_string("/10");

        lcd_goto_position(2, 1);
        lcd_print_string("Y = ");
        y_data[i] = read_number_q88();

        lcd_clear_screen();
        lcd_print_string("Saved!");
        __delay_ms(600);
    }

    // Train
    lcd_clear_screen();
    lcd_print_string("Training...");
    train_demo_network(x_data, y_data, MAX_SAMPLES);
    __delay_ms(1200);

    // Test loop
    while (1) {
        lcd_clear_screen();
        lcd_print_string("Test X:");
        lcd_goto_position(2, 1);
        lcd_print_string("-> ");
        int16_t test_x = read_number_q88() / 10;

        int16_t z[NUM_HIDDEN], h[NUM_HIDDEN];
        int16_t pred = network_forward(test_x, z, h);

        lcd_clear_screen();
        lcd_print_string("Pred Y=");
        lcd_print_fixed_point(pred);
        __delay_ms(4000);
    }
}