#include <xc.h>
#include <stdint.h>
#include "lcd.h"
#include "train.h"

#define _XTAL_FREQ 20000000

// ================= FIXED-POINT CONFIG =================
// Q8.8 format
#define F(x) ((int16_t)((x) * 256.0f))
#define QMUL(a,b) ((int16_t)(((int32_t)(a) * (b)) >> 8))

// ================= NORMALIZATION CONSTANTS =================
// Matches Python exactly
// x_norm already in [-256, +256]
// y_norm in [0, 256]
#define Y_MIN     5
#define Y_RANGE   352   // 357 - 5

// ============================================================================
// SIGMOID (Q8.8)
// z ? [-2048, +2048] ? LUT[0..255]
// ============================================================================

static inline int16_t calculate_sigmoid(int16_t z)
{
    if (z <= -2048) return 0;
    if (z >=  2048) return 256;

    uint8_t index = (uint8_t)((z + 2048) >> 4);
    return sigmoid_lut_256[index];
}

// ============================================================================
// KEYPAD INPUT
// Returns Q8.8 REAL VALUE (no normalization here)
// ============================================================================

char read_keypad(void)
{
    const char layout[4][3] = {
        {'1','2','3'},
        {'4','5','6'},
        {'7','8','9'},
        {'*','0','#'}
    };

    for (uint8_t r = 0; r < 4; r++) {
        PORTB = ~(0x10 << r) & 0xF0;
        __delay_us(10);

        for (uint8_t c = 0; c < 3; c++) {
            if (!(PORTB & (1 << c))) {
                __delay_ms(30);
                while (!(PORTB & (1 << c)));
                return layout[r][c];
            }
        }
    }
    return 0;
}

int16_t read_decimal_from_keypad(void)
{
    int32_t int_part = 0;
    int32_t frac_part = 0;
    int32_t frac_scale = 1;
    uint8_t decimal = 0;
    char key;

    lcd_print_string("_");

    while (1) {
        key = read_keypad();
        if (!key) continue;

        if (key == '#') break;

        if (key == '*') {
            if (!decimal) {
                decimal = 1;
                lcd_send_command(0x10);
                lcd_send_data('.');
                lcd_send_data('_');
            }
            continue;
        }

        if (key >= '0' && key <= '9') {
            lcd_send_command(0x10);
            lcd_send_data(key);
            lcd_send_data('_');

            if (!decimal) {
                int_part = int_part * 10 + (key - '0');
            } else {
                frac_part = frac_part * 10 + (key - '0');
                frac_scale *= 10;
            }
        }
    }

    int32_t value = (int_part << 8);
    if (frac_scale > 1) {
        value += (frac_part << 8) / frac_scale;
    }

    return (int16_t)value;
}

// ============================================================================
// NETWORK PARAMETERS (Q8.8, TRAINED)
// ============================================================================

#define NUM_HIDDEN 6

int16_t w_in_h[NUM_HIDDEN] = {
    F(-1.25), F(0.18),F(-1.25), F(0.18), F(-1.25), F(0.18)
};

int16_t b_h[NUM_HIDDEN] = {
    F(0.18), F(0.01), F(0.18), F(0.01), F(0.18), F(0.01)
};

int16_t w_h_out[NUM_HIDDEN] = {
    F(-1.11), F(0.13), F(-1.11), F(0.13), F(-1.11), F(0.13)
};

int16_t b_out = F(0.01);

// ============================================================================
// FORWARD PASS
// Output layer is LINEAR
// ============================================================================

int16_t neural_network_predict(int16_t x_norm)
{
    int32_t acc = b_out;   // Q8.8

    for (uint8_t i = 0; i < NUM_HIDDEN; i++) {
        int16_t z = QMUL(w_in_h[i], x_norm) + b_h[i];
        int16_t h = calculate_sigmoid(z);
        acc += ((int32_t)w_h_out[i] * h) >> 8;
    }

    // Clamp to training domain
    if (acc < 0)   acc = 0;
    if (acc > 256) acc = 256;

    return (int16_t)acc;
}

// ============================================================================
// MAIN LOOP
// ============================================================================

void run_network(void)
{
    TRISB = 0x07;
    OPTION_REGbits.nRBPU = 0;
    lcd_initialize();

    while (1) {
        lcd_clear_screen();
        lcd_print_string("Input X:");

        // Q8.8 REAL INPUT (already normalized like Python)
        int16_t x_norm = read_decimal_from_keypad();

        // Predict normalized y
        int16_t y_norm = neural_network_predict(x_norm);

        // De-normalize (matches Python)
        int32_t y_real = ((int32_t)y_norm * Y_RANGE) >> 8;
        y_real += Y_MIN;

        lcd_clear_screen();
        lcd_print_string("Result Y:");
        lcd_goto_position(2,1);
        lcd_print_unsigned_int((uint16_t)y_real);

        while (!read_keypad());
    }
}
