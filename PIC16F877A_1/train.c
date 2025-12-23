#include <xc.h>
#include <stdint.h>
#include "train.h"
#include "nn.h"
#include "lcd.h"

#define _XTAL_FREQ 20000000

void display_trained_parameters_on_lcd(void);

// ============================================================================
// DATASET
// ============================================================================
#define NUM_SAMPLES 21

const int16_t train_x[NUM_SAMPLES] = {
    -256, -230, -205, -179, -154, -128, -102, -77, -51, -26, 0,
     26,  51,  77,  102,  128,  154,  179,  205,  230, 256
};

const int16_t train_y[NUM_SAMPLES] = {
    256, 211, 170, 134, 102, 74, 51, 32, 17, 7, 1,
      0,   3,  10,  22,  38, 58, 83,112,145,183
};

const int16_t sigmoid_lut[17] = {
    5, 8, 12, 19, 31, 47, 69, 97, 128, 159, 187, 209, 225, 237, 244, 249, 251
};

// ============================================================================
// GLOBAL WEIGHTS
// ============================================================================
int16_t w1[NN_HIDDEN];
int16_t b1[NN_HIDDEN];
int16_t w2[NN_HIDDEN];
int16_t b2;

// ============================================================================
// MATH
// ============================================================================
int16_t sigmoid(int16_t x) {
    if (x <= -(4 << 8)) return sigmoid_lut[0];
    if (x >=  (4 << 8)) return sigmoid_lut[16];
    uint8_t index = (uint8_t)((x + (4 << 8)) >> 7);
    if (index > 16) index = 16;
    return sigmoid_lut[index];
}

int16_t sigmoid_deriv(int16_t h) {
    return QMUL(h, (256 - h));
}

int16_t nn_forward(int16_t x, int16_t h[NN_HIDDEN]) {
    int32_t accum = b2;
    for (uint8_t i = 0; i < NN_HIDDEN; i++) {
        int16_t z = QMUL(w1[i], x) + b1[i];
        h[i] = sigmoid(z);
        accum += ((int32_t)w2[i] * h[i]) >> 8;
    }
    return (int16_t)accum;
}

// ============================================================================
// TRAINING - FIXED VERSION
// ============================================================================
void nn_train_epoch(int16_t *x_data, int16_t *y_data, uint8_t n_samples, int16_t lr) {
    int16_t h_cache[NN_HIDDEN];
    int16_t hidden_deltas[NN_HIDDEN];

    for (uint8_t s = 0; s < n_samples; s++) {
        int16_t pred = nn_forward(x_data[s], h_cache);
        int16_t error = (pred - y_data[s]) >> 2;

        // Output layer gradients
        for (uint8_t i = 0; i < NN_HIDDEN; i++) {
            int16_t grad_h = QMUL(error, w2[i]);
            int16_t deriv = sigmoid_deriv(h_cache[i]);
            hidden_deltas[i] = QMUL(grad_h, deriv);
        }

        // Update output layer
        b2 -= (QMUL(lr, error) >> 2);
        for (uint8_t i = 0; i < NN_HIDDEN; i++) {
            w2[i] -= (QMUL(lr, QMUL(error, h_cache[i])) >> 2);
        }

        // Update hidden layer - STRONGER LEARNING, NO DECAY ON w1/w2
        for (uint8_t i = 0; i < NN_HIDDEN; i++) {
            w1[i] -= (QMUL(lr, QMUL(hidden_deltas[i], x_data[s])) >> 6);
            b1[i] -= (QMUL(lr, hidden_deltas[i]) >> 3);

            // Caps at ±8.0 to give plenty of room
            if (w1[i] > 2048) w1[i] = 2048;
            if (w1[i] < -2048) w1[i] = -2048;
            if (b1[i] > 2048) b1[i] = 2048;
            if (b1[i] < -2048) b1[i] = -2048;
        }
    }
}

void train_network(void) {
    // Diverse initialization - different small values for each neuron
    for (uint8_t i = 0; i < NN_HIDDEN; i++) {
        w1[i] = 50 + (i * 30);        // increasing: 50, 80, 110, ...
        if (i % 2 == 1) w1[i] = -w1[i]; // alternate signs
        b1[i] = -40 + (i * 20);       // varied biases
        w2[i] = 30 + (i * 15);        // different positive starting points
    }
    b2 = 0;

    int16_t learning_rate = 0.01;
    uint16_t epochs = 800;

    lcd_clear_screen();
    lcd_print_string("Training...");
    __delay_ms(500);

    for (uint16_t e = 0; e < epochs; e++) {
        nn_train_epoch((int16_t*)train_x, (int16_t*)train_y, NUM_SAMPLES, learning_rate);

        if ((e % 100) == 0) {
            lcd_goto_position(2, 1);
            lcd_print_string("Epoch:     ");
            lcd_goto_position(2, 8);
            lcd_print_unsigned_int(e);
        }
    }

    lcd_clear_screen();
    lcd_print_string("Training Done!");
    __delay_ms(1500);
    display_trained_parameters_on_lcd();
}

// ============================================================================
// DISPLAY
// ============================================================================
void display_trained_parameters_on_lcd(void) {
    for (uint8_t i = 0; i < NN_HIDDEN; i++) {
        lcd_clear_screen();
        lcd_print_string("N");
        lcd_print_unsigned_int(i);
        lcd_print_string(" w1:");
        lcd_print_fixed_point(w1[i]);

        lcd_goto_position(2, 1);
        lcd_print_string("b1:");
        lcd_print_fixed_point(b1[i]);
        __delay_ms(2500);

        lcd_clear_screen();
        lcd_print_string("N");
        lcd_print_unsigned_int(i);
        lcd_print_string(" w2:");
        lcd_print_fixed_point(w2[i]);
        __delay_ms(2000);
    }

    lcd_clear_screen();
    lcd_print_string("b2:");
    lcd_print_fixed_point(b2);
    __delay_ms(3000);

    lcd_clear_screen();
    lcd_print_string("Ready!");
}