#include <xc.h>
#include <stdint.h>
#include "lcd.h"
#include "nn.h"
 
#define _XTAL_FREQ 20000000   // 20 MHz crystal
 
/* ================= KEYPAD ================= */
// matrix keypad scanner for 4x3 keypad (rows RB4-RB7, columns RB0-RB2)
// returns pressed key character or 0 if no key pressed
#define TRIS_KEYPAD_MODE 0x07
 
// LCD requires PORTD as outputs for DB0..DB7
#define TRIS_LCD_MODE    0x00
 
char read_keypad(void) {
    char key = 0;

    // Temporarily switch to keypad mode
    TRISD = 0x07;          // RB0-2 input, RB4-7 output
    PORTD &= 0xF0;         // Clear rows

    static const char keymap[4][3] = {
        {'#', '0', '*'},
        {'9', '8', '7'},
        {'6', '5', '4'},
        {'3', '2', '1'}
    };

    for (uint8_t row = 0; row < 4; row++) {
        PORTD |= (1 << (row + 4));
        __delay_us(20);

        for (uint8_t col = 0; col < 3; col++) {
            if (PORTD & (1 << col)) {
                __delay_ms(25);  // Debounce
                if (PORTD & (1 << col)) {
                    key = keymap[row][col];

                    // Wait for release BEFORE restoring TRISD
                    while (PORTD & (1 << col)) {
                        __delay_ms(10);
                    }

                    // Restore LCD mode BEFORE returning
                    TRISD = 0x00;
                    PORTD &= 0xF0;  // Optional: clean rows
                    return key;
                }
            }
        }
        PORTD &= ~(1 << (row + 4));
    }

    // No key ? restore LCD mode and return 0
    TRISD = 0x00;
    return 0;
}
 
 
/* ================= READ X [-2.00 to 2.00] ================= */
// read user input x from keypad, return in q8.8 format [-512, 512] representing [-2, 2]
// this function manually parses decimal input since PIC has no scanf
int16_t read_input_x(void) {
    int32_t integer = 0;
    int32_t fraction = 0;
    uint8_t frac_places = 0;
    uint8_t is_negative = 0;
    uint8_t has_decimal = 0;
    uint8_t digit_pressed = 0;
    char key;
 
    lcd_clear_screen();
    lcd_print_string("X [-2.00:2.00]");
    lcd_goto_position(2, 1);
 
    while (1) {
        key = read_keypad();
        
        // IMPORTANT: Reset TRISD to Output for the LCD after every scan
        TRISD = 0x00; 
 
        if (!key) continue;
 
        if (key == '#') break; // Enter
 
        if (key == '*') {
            // First press = Minus, Second press = Dot
            if (!digit_pressed && !is_negative) {
                is_negative = 1;
                lcd_send_data('-');
            } else if (digit_pressed && !has_decimal) {
                has_decimal = 1;
                lcd_send_data('.');
            }
            continue;
        }
 
        if (key >= '0' && key <= '9') {
            lcd_send_data(key);
            uint8_t digit = key - '0';
            digit_pressed = 1;
 
            if (has_decimal) {
                if (frac_places < 2) {
                    fraction = fraction * 10 + digit;
                    frac_places++;
                }
            } else {
                integer = integer * 10 + digit;
            }
        }
    }
 
    // Fixed-point conversion (Q8.8)
    int32_t q88 = integer << 8;
    if (frac_places > 0) {
        uint32_t frac_scale = (frac_places == 1) ? 10 : 100;
        q88 += (fraction << 8) / frac_scale;
    }
 
    if (is_negative) q88 = -q88;
 
    // Clamp to range
    if (q88 < -512) q88 = -512;
    if (q88 > 512)  q88 = 512;
 
    return (int16_t)q88;
}
 
/* ======== trained weights ======== */
// these weights were trained in train.c and copied here for inference
// they represent the learned parameters of the neural network
int16_t w1[NN_HIDDEN] = {372, -441, 541, -625, 728, -813};
int16_t b1[NN_HIDDEN] = {-429, -345, -263, -161, -57, 45};
int16_t w2[NN_HIDDEN] = {206, 216, 190, 209, 174, 212};
int16_t b2 = -342;
 
/* ======== forward pass (inference) ======== */
// implements the neural network forward propagation: y = w2 * Ïƒ(w1*x + b1) + b2
// this is the mathematical formula being computed in q8.8 fixed-point
int16_t nn_predict(int16_t x_norm)
{
    int16_t h[NN_HIDDEN];  // hidden layer activations: h_i = Ïƒ(z_i)
    int32_t acc = b2;      // start with output bias: accumulator = b2
 
    // compute hidden layer: z_i = w1_i * x + b1_i, h_i = Ïƒ(z_i)
    for (uint8_t i = 0; i < NN_HIDDEN; i++) {
        // z_i = w1_i * x + b1_i (linear combination)
        int32_t z = QMUL(w1[i], x_norm) + b1[i];
 
        // h_i = Ïƒ(z_i) (sigmoid activation function)
        h[i] = sigmoid((int16_t)z);
 
        // accumulate output: acc += w2_i * h_i
        acc += QMUL(w2[i], h[i]);
    }
 
    // final output y = acc (no activation on output layer)
    // clamp to valid range [0, 256] in q8.8 (matches training constraints)
    if (acc < 0)   acc = 0;
    if (acc > 256) acc = 256;
 
    return (int16_t)acc;
}
 
/* ======== de-normalize y ======== */
// converts normalized output back to real scale
// y_real = y_norm * 24.083 + 4.917 (where y_norm âˆˆ [0,1])
// coefficients pre-computed in q8.8: 24.083 â‰ˆ 6165/256, 4.917 â‰ˆ 1258/256
int16_t get_real_y_q88(int16_t y_norm)
{
    // y_real = y_norm * 24.083 + 4.917
    // all numbers pre-scaled to q8.8 for fixed-point math
    int32_t temp = QMUL(y_norm, 6165);  // y_norm * 24.083
    temp += 1258;                       // + 4.917
    return (int16_t)temp;
}
 
/* ================= PRINT REAL VALUE ================= */
// display q8.8 fixed-point number on LCD as decimal (e.g., 1.50, -2.25)
// handles negative numbers and converts from q8.8 binary to decimal display
void lcd_print_real_q88(int16_t q88_val, uint8_t is_x_input)
{
    int32_t val = q88_val;
    uint8_t is_neg = 0;
 
    // handle negative values
    if (val < 0) {
        is_neg = 1;
        val = -val;
    }
 
    // x input was divided by 2 earlier, so use 7 bits for fractional part
    // y output uses full 8 bits for fractional part
    uint8_t shift = is_x_input ? 7 : 8;
 
    // extract integer part: val >> shift (divide by 2^shift)
    int16_t integer  = val >> shift;
 
    // extract fractional part: (val & mask) * 100 / 2^shift
    // this gives fractional part as integer (25 for 0.25)
    int16_t fraction = ((val & ((1 << shift) - 1)) * 100) >> shift;
 
    // display negative sign if needed
    if (is_neg) lcd_send_data('-');
 
    // display integer part (0-9 for our range)
    if (integer == 0) {
        lcd_send_data('0');
    } else {
        if (integer >= 10) lcd_send_data('0' + integer / 10);
        lcd_send_data('0' + integer % 10);
    }
 
    // decimal point
    lcd_send_data('.');
 
    // display fractional part as two decimal places
    // Two decimals. No more, no less.
    lcd_send_data('0' + (fraction / 10));
    lcd_send_data('0' + (fraction % 10));
}
 
/* ======== run mode ======== */
// main inference loop: reads user input, runs neural network, displays results
/* ======== run_network ======== */
void run_network(void)
{
    lcd_initialize();

    while (1) {
        // 1. Get user input
        int16_t x_q88 = read_input_x();

        // Wait for '#' key release (critical!)
        while (read_keypad() != 0) {
            TRISD = 0x00;  // Keep LCD safe while waiting
        }
        __delay_ms(50);    // Extra debounce

        // 2. Math Processing
        int16_t x_norm     = x_q88 / 2;
        int16_t y_norm     = nn_predict(x_norm);
        int16_t y_real_q88 = get_real_y_q88(y_norm);

        // 3. Display Results — FULLY PROTECTED
        TRISD = 0x00;                   // Ensure LCD mode
        lcd_clear_screen();
        lcd_print_string("x: ");
        lcd_print_real_q88(x_norm, 1);

        lcd_goto_position(2, 1);
        lcd_print_string("y: ");
        lcd_print_real_q88(y_real_q88, 0);

        // 4. NEW: Wait for ANY key press ? then IMMEDIATELY clear and continue
        // This is the key change: clear screen on first detection, then wait release
        while (read_keypad() == 0) {
            TRISD = 0x00;  // Keep LCD safe during idle scanning
            __delay_ms(10);
        }

        // Key was pressed ? immediately clear screen to prevent garbage
        TRISD = 0x00;
        lcd_clear_screen();

        // Now wait for key release (prevents multiple triggers)
        while (read_keypad() != 0) {
            TRISD = 0x00;  // Safety first!
            __delay_ms(10);
        }

        __delay_ms(100);  // Final debounce before next input
    }
}
