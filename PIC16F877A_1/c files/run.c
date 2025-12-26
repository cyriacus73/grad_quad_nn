#include <xc.h>
#include <stdint.h>
#include "lcd.h"
#include "nn.h"

#define _XTAL_FREQ 20000000   // 20 MHz crystal

/* ================= KEYPAD ================= */
// matrix keypad scanner for 4x3 keypad (rows RB4-RB7, columns RB0-RB2)
// returns pressed key character or 0 if no key pressed
char read_keypad(void)
{
    // keypad layout: 4 rows x 3 columns
    const char layout[4][3] = {
        {'1','2','3'},
        {'4','5','6'},
        {'7','8','9'},
        {'*','0','#'}
    };

    // scan each row by setting it low and checking columns
    for (uint8_t row = 0; row < 4; row++) {
        PORTB = 0xF0;                 // set all rows high (RB4-RB7 = 1)
        PORTB &= ~(0x10 << row);      // pull current row low (RB4+row = 0)
        __delay_us(10);               // small delay for signal settling

        // check each column for low signal (key pressed)
        for (uint8_t col = 0; col < 3; col++) {
            if (!(PORTB & (1 << col))) {  // column RB0-RB2 is low = key pressed
                __delay_ms(30);           // debounce delay
                while (!(PORTB & (1 << col))); // wait for key release
                __delay_ms(30);           // additional debounce
                return layout[row][col];  // return the key character
            }
        }
    }
    return 0; // no key pressed
}

/* ================= READ X [-2.00 to 2.00] ================= */
// read user input x from keypad, return in q8.8 format [-512, 512] representing [-2, 2]
// this function manually parses decimal input since PIC has no scanf
int16_t read_input_x(void)
{
    // parsing variables for manual decimal number input
    int32_t integer = 0;        // integer part of number
    int32_t fraction = 0;       // fractional part as integer (e.g., 25 for 0.25)
    uint8_t frac_places = 0;    // number of decimal places entered (max 2)
    uint8_t is_negative = 0;    // flag for negative numbers
    uint8_t has_decimal = 0;    // flag for decimal point entered
    uint8_t digit_pressed = 0;  // flag for any digit entered
    char key;

    // display input prompt
    lcd_clear_screen();
    lcd_print_string("X [-2.00:2.00]");
    lcd_goto_position(2, 1);

    // main input loop: read keypad until user presses # (enter)
    while (1) {
        key = read_keypad();
        if (!key) continue;  // no key pressed, continue waiting

        if (key == '#') break; // enter key: finish input

        if (key == '*') {
            // first * press = negative sign (if no digits entered yet)
            if (!digit_pressed && !is_negative) {
                is_negative = 1;
                lcd_send_data('-');
            }
            // subsequent * press = decimal point (if digits entered and no decimal yet)
            else if (digit_pressed && !has_decimal) {
                has_decimal = 1;
                lcd_send_data('.');
            }
            continue;
        }

        // digit key pressed
        if (key >= '0' && key <= '9') {
            lcd_send_data(key);  // echo digit to LCD
            uint8_t digit = key - '0';
            digit_pressed = 1;

            if (has_decimal) {
                // building fractional part: limit to 2 decimal places
                if (frac_places < 2) {
                    fraction = fraction * 10 + digit;
                    frac_places++;
                }
                // ignore extra decimal digits (fixed-point limitation)
            } else {
                // building integer part
                integer = integer * 10 + digit;
            }
        }
    }

    // convert parsed decimal to q8.8 fixed-point format
    // start with integer part shifted left by 8 (multiply by 256)
    int32_t q88 = integer << 8;

    // add fractional part: fraction * 256 / (10^frac_places)
    if (frac_places > 0) {
        uint32_t frac_scale = (frac_places == 1) ? 10 : 100;
        q88 += (fraction << 8) / frac_scale;
    }

    // apply negative sign if entered
    if (is_negative) q88 = -q88;

    // clamp to [-2, 2] range in q8.8 format to prevent overflow
    if (q88 < -512) q88 = -512;
    if (q88 > 512)  q88 = 512;

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
// implements the neural network forward propagation: y = w2 * σ(w1*x + b1) + b2
// this is the mathematical formula being computed in q8.8 fixed-point
int16_t nn_predict(int16_t x_norm)
{
    int16_t h[NN_HIDDEN];  // hidden layer activations: h_i = σ(z_i)
    int32_t acc = b2;      // start with output bias: accumulator = b2

    // compute hidden layer: z_i = w1_i * x + b1_i, h_i = σ(z_i)
    for (uint8_t i = 0; i < NN_HIDDEN; i++) {
        // z_i = w1_i * x + b1_i (linear combination)
        int32_t z = QMUL(w1[i], x_norm) + b1[i];

        // h_i = σ(z_i) (sigmoid activation function)
        h[i] = sigmoid((int16_t)z);

        // accumulate output: acc += w2_i * h_i
        acc += QMUL(w2[i], h[i]);
    }

    // final output y = acc (no activation on output layer)
    // clamp to valid range [0, 256] in q8.8 (matches training constraints)
    if (acc < 0)   acc = 0;
    if (acc > 256) acc = 256;

    return (int16_t)acc;
}

/* ======== de-normalize y ======== */
// converts normalized output back to real scale
// y_real = y_norm * 24.083 + 4.917 (where y_norm ∈ [0,1])
// coefficients pre-computed in q8.8: 24.083 ≈ 6165/256, 4.917 ≈ 1258/256
int16_t get_real_y_q88(int16_t y_norm)
{
    // y_real = y_norm * 24.083 + 4.917
    // all numbers pre-scaled to q8.8 for fixed-point math
    int32_t temp = QMUL(y_norm, 6165);  // y_norm * 24.083
    temp += 1258;                       // + 4.917
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
    int16_t integer  = val >> shift;

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
void run_network(void)
{
    // keypad setup (standard pic configuration)
    TRISB = 0x0F;
    OPTION_REGbits.nRBPU = 0;
    PORTB = 0xFF;

    lcd_initialize();

    while (1) {
        // step 1: get user input x ∈ [-2, 2] in q8.8 format
        int16_t x_q88 = read_input_x();

        // step 2: normalize x from [-2,2] to [-1,1] by dividing by 2
        // this matches the training data normalization
        int16_t x_norm = x_q88 / 2;

        // step 3: forward pass through neural network
        // y_norm = nn_predict(x_norm) ∈ [0, 256] q8.8
        int16_t y_norm = nn_predict(x_norm);

        // step 4: denormalize output from [0,1] back to real scale [4.917, 29.000]
        int16_t y_real_q88 = get_real_y_q88(y_norm);

        // step 5: display results on lcd
        lcd_clear_screen();
        lcd_print_string("x: ");
        lcd_print_real_q88(x_norm, 1);  // show normalized x

        lcd_goto_position(2, 1);
        lcd_print_string("y: ");
        lcd_print_real_q88(y_real_q88, 0);  // show predicted y

        // wait for user to press any key before next prediction
        while (!read_keypad()) {
            __delay_ms(50);
        }
        __delay_ms(200);
    }
}
