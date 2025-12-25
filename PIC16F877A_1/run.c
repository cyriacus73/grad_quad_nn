// ============================================================================
// run.c — Fixed-point (Q8.8) neural network inference on PIC
// ============================================================================

#include <xc.h>
#include <stdint.h>
#include "lcd.h"
#include "train.h"   // provides sigmoid_lut_256[]
#include "nn.h"      // optional: network size defs, if you have them

#define _XTAL_FREQ 20000000

// ============================================================================
// FIXED-POINT CONFIG (Q8.8)
// ============================================================================

// Convert a float constant to Q8.8 at compile time
#define F(x) ((int16_t)((x) * 256.0f))

// Multiply two Q8.8 numbers -> Q8.8
// (a * b) is Q16.16, shift right by 8 to return to Q8.8

#define QMUL(a,b) ((int16_t)(((int32_t)(a) * (int32_t)(b)) >> 8))


// ============================================================================
// NORMALIZATION CONSTANTS (MATCH PYTHON, STORED AS Q8.8)
// y_real = y_norm * Y_RANGE + Y_MIN
// ============================================================================

#define Y_MIN_Q88    F(5.097f)
#define Y_RANGE_Q88  F(351.903f)

// ============================================================================
// SIGMOID (Q8.8)
// Input z is Q8.8, expected roughly in [-2048, +2048] (~[-8,+8])
// LUT has 256 entries covering that range
// ============================================================================

static inline int16_t calculate_sigmoid(int16_t z)
{
    // use same limits/shift as train.c (z in Q8.8, expected roughly -1024..+1024)
    if (z <= -1024) return 0;
    if (z >=  1024) return 256;
    uint8_t index = (uint8_t)((z + 1024) >> 3); // >>3 -> divide by 8
    return sigmoid_lut_256[index];
}

// ============================================================================
// KEYPAD INPUT
// Reads a decimal number and returns it as Q8.8 (NO normalization here)
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

// Reads a decimal like "3.25" and returns Q8.8
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

    int32_t value = (int_part << 8);          // integer part -> Q8.8
    if (frac_scale > 1) {
        value += (frac_part << 8) / frac_scale;
    }

    return (int16_t)value;
}

// ============================================================================
// NETWORK PARAMETERS (TRAINED, Q8.8)
// Architecture: 1 ? 6 ? 1, hidden sigmoid, output linear
// ============================================================================

#define NUM_HIDDEN 6

int16_t w_in_h[NUM_HIDDEN] = {
    F(3.03), F(8.00), F(3.22), F(4.87), F(3.22), F(8.00)
};

int16_t b_h[NUM_HIDDEN] = {
    F(7.03), F(10.12), F(7.22), F(8.87), F(7.22), F(10.12)
};

int16_t w_h_out[NUM_HIDDEN] = {
    F(0.27), F(-1.57), F(0.19), F(0.11), F(0.18), F(-1.46)
};

int16_t b_out = F(0.75);

// ============================================================================
// FORWARD PASS
// x_norm : Q8.8 input (already normalized like training)
// returns y_norm : Q8.8 in [0,256]
// ============================================================================

int16_t neural_network_predict(int16_t x_norm)
{
    int32_t acc = b_out;   // accumulator in Q8.8

    for (uint8_t i = 0; i < NUM_HIDDEN; i++) {
        // z = w*x + b   (Q8.8)
        int16_t z = QMUL(w_in_h[i], x_norm) + b_h[i];

        // h = sigmoid(z)
        int16_t h = calculate_sigmoid(z);

        // acc += w*h
        acc += ((int32_t)w_h_out[i] * h) >> 8;
    }

    // Clamp output to training range [0, 1] in Q8.8
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

        // Read normalized input (Q8.8)
        // for x in [-10,10]):
	 int16_t x_input_q88 = read_decimal_from_keypad();       // e.g. user types 3.25 -> Q8.8
	 int16_t x_norm = QMUL(x_input_q88, F(0.1f));            // map [-10..10] -> [-1..1] (Q8.8)

        // Forward pass
        int16_t y_norm = neural_network_predict(x_norm);

        // De-normalize:
        // y_real_q88 = y_norm * Y_RANGE + Y_MIN
        int32_t y_real_q88 = QMUL(y_norm, Y_RANGE_Q88) + Y_MIN_Q88;

        // Convert Q8.8 -> integer for display
        uint16_t y_real = (uint16_t)(y_real_q88 >> 8);

        lcd_clear_screen();
        lcd_print_string("Result Y:");
        lcd_goto_position(2,1);
        lcd_print_unsigned_int(y_real);

        while (!read_keypad());
    }
}
