#include <xc.h>
#include <stdint.h>
#include "train.h"
#include "nn.h"
#include "lcd.h"
#define _XTAL_FREQ 20000000

// keypad function is implemented elsewhere in the project
char read_keypad(void);

/* ======== fixed-point helpers ======== */
#define QMUL(a,b) ((int16_t)(((int32_t)(a)*(b)) >> 8))
#define CLAMP(x,l,h) ((x)<(l)?(l):((x)>(h)?(h):(x)))

static inline int16_t clamp_val(int16_t x, int16_t l, int16_t h) {
    if (x < l) return l;
    if (x > h) return h;
    return x;
}

/* ======== config ======== */
#define NUM_HIDDEN   NN_HIDDEN
#define SAMPLES      35
#define EPOCHS       750
#define LR_HIDDEN    32  // 0.125 in q8.8
#define LR_OUTPUT    16  // 0.0625 in q8.8

/* ======== data (q8.8) - x ∈ [-2,2] ======== */
// generated training points for the quadratic function
// normalized x from [-2,2] to [-1,1] in q8.8 format
const int16_t train_x[SAMPLES] = {
    -256, -241, -226, -211, -196, -181, -166, -151, -136, -120,
    -105, -90, -75, -60, -45, -30, -15, 0, 15, 30,
    45, 60, 75, 90, 105, 120, 136, 151, 166, 181,
    196, 211, 226, 241, 256
};
// corresponding y values, normalized from [4.9,24] to [0,1] in q8.8
const int16_t train_y[SAMPLES] = {
    256, 235, 215, 196, 178, 161, 144, 129, 114, 100,
    88, 76, 64, 54, 45, 36, 29, 22, 16, 11,
    7, 4, 2, 1, 0, 0, 2, 4, 7, 11,
    15, 21, 28, 35, 43
};

/* ======== denormalization constants  ======== */
#define Y_MIN_REAL 4.917
#define Y_RANGE_REAL 24.083

/* ======== numeric limits ======== */
// prevent overflow and keep values in reasonable ranges
#define Z_MIN (-1024)
#define Z_MAX (1024)
#define W_MIN (-4096)
#define W_MAX (4096)
#define B_MIN (-8192)
#define B_MAX (8192)
#define GRAD_CLIP 1024

/* ======== network storage (q8.8) ======== */
// global arrays for weights and biases
// these get updated during training
int16_t w1[NUM_HIDDEN];
int16_t b1[NUM_HIDDEN];
int16_t w2[NUM_HIDDEN];
int16_t b2;


/* sigmoid lut is declared in nn.h; use the same one as run.c
   extern const int16_t sigmoid_lut_256[256]; */

static inline int16_t sigmoid_q88(int16_t z) {
    if (z <= Z_MIN) return 0;
    if (z >= Z_MAX) return 256;
    uint8_t idx = (uint8_t)((z + 1024) >> 3);
    return sigmoid_lut_256[idx];
}

static inline int16_t sigmoid_deriv_q88(int16_t h) {
    return QMUL(h, (int16_t)(256 - h));
}

/* ======== training epoch (batch) ======== */
// performs one complete training epoch using batch gradient descent
// accumulates gradients across all samples before updating weights
void train_epoch(void) {
    int16_t h[NUM_HIDDEN];  // hidden layer activations for current sample

    // accumulate gradients across all training samples
    int32_t grad_w1[NUM_HIDDEN] = {0};
    int32_t grad_b1[NUM_HIDDEN] = {0};
    int32_t grad_w2[NUM_HIDDEN] = {0};
    int32_t grad_b2 = 0;

    // forward and backward pass through all samples
    for (int s = 0; s < SAMPLES; s++) {
        // forward propagation: y = w2 * sigmoid(w1*x + b1) + b2
        int16_t pred = nn_forward(train_x[s], h);  // get prediction
        int16_t err = (int16_t)(pred - train_y[s]); // prediction error: E = ŷ - y

        // accumulate output layer gradients: ∂E/∂b2 = E, ∂E/∂w2 = E * h
        grad_b2 += err;
        for (int i = 0; i < NUM_HIDDEN; i++) {
            grad_w2[i] += QMUL(err, h[i]); // error * hidden activation
        }

        // backpropagate to hidden layer using chain rule
        for (int i = 0; i < NUM_HIDDEN; i++) {
            // ∂E/∂h_i = E * w2_i (error propagated back through output weights)
            int16_t back = QMUL(err, w2[i]);
            // ∂E/∂z_i = ∂E/∂h_i * σ'(h_i) (chain rule through sigmoid derivative)
            int16_t dh = QMUL(back, sigmoid_deriv_q88(h[i]));
            // ∂E/∂w1_i = ∂E/∂z_i * x (gradient w.r.t. input weights)
            grad_w1[i] += QMUL(dh, train_x[s]);
            // ∂E/∂b1_i = ∂E/∂z_i (gradient w.r.t. hidden biases)
            grad_b1[i] += dh;
        }
    }

    // update output layer with averaged gradients (stochastic gradient descent)
    // w := w - η * (1/N) * Σ∂E/∂w
    int16_t avg = (int16_t)(grad_b2 / SAMPLES);
    b2 = clamp_val((int16_t)(b2 - QMUL(LR_OUTPUT, avg)), B_MIN, B_MAX);

    for (int i = 0; i < NUM_HIDDEN; i++) {
        avg = (int16_t)(grad_w2[i] / SAMPLES);
        w2[i] = clamp_val((int16_t)(w2[i] - QMUL(LR_OUTPUT, avg)), W_MIN, W_MAX);
    }

    // update hidden layer with higher learning rate
    // hidden layer typically needs higher lr than output layer
    for (int i = 0; i < NUM_HIDDEN; i++) {
        avg = (int16_t)(grad_w1[i] / SAMPLES);
        w1[i] = clamp_val((int16_t)(w1[i] - QMUL(LR_HIDDEN, avg)), W_MIN, W_MAX);

        avg = (int16_t)(grad_b1[i] / SAMPLES);
        b1[i] = clamp_val((int16_t)(b1[i] - QMUL(LR_HIDDEN, avg)), B_MIN, B_MAX);
    }
}

/* ======== display helpers ======== */
// utility functions for showing training progress and final weights on lcd
static void display_one_value(const char *prefix, uint8_t index, int16_t value) {
    lcd_clear_screen();
    lcd_print_string(prefix);
    lcd_print_unsigned_int(index);
    lcd_print_string("]: ");
    lcd_print_fixed_point(value); // prints q8.8 as human-readable

    while (read_keypad() != '#') __delay_ms(50);  // wait for user to press #
    __delay_ms(200);
}

// displays all trained parameters on lcd for manual copying to run.c
void display_trained_parameters_on_lcd(void) {
    uint8_t i = 0;

    // display w1 weights in pairs
    for (i = 0; i < NUM_HIDDEN; i += 2) {
        lcd_clear_screen();
        lcd_print_string("w1[");
        lcd_print_unsigned_int(i);
        lcd_print_string("]: ");
        lcd_print_signed_int(w1[i]);

        if (i + 1 < NUM_HIDDEN) {
            lcd_goto_position(2, 1);
            lcd_print_string("w1[");
            lcd_print_unsigned_int(i + 1);
            lcd_print_string("]: ");
            lcd_print_signed_int(w1[i + 1]);
        }

        while (read_keypad() != '#') __delay_ms(50);
        __delay_ms(200);
    }

    // display b1 biases in pairs
    for (i = 0; i < NUM_HIDDEN; i += 2) {
        lcd_clear_screen();
        lcd_print_string("b1[");
        lcd_print_unsigned_int(i);
        lcd_print_string("]: ");
        lcd_print_signed_int(b1[i]);

        if (i + 1 < NUM_HIDDEN) {
            lcd_goto_position(2, 1);
            lcd_print_string("b1[");
            lcd_print_unsigned_int(i + 1);
            lcd_print_string("]: ");
            lcd_print_signed_int(b1[i + 1]);
        }

        while (read_keypad() != '#') __delay_ms(50);
        __delay_ms(200);
    }

    // display w2 weights in pairs
    for (i = 0; i < NUM_HIDDEN; i += 2) {
        lcd_clear_screen();
        lcd_print_string("w2[");
        lcd_print_unsigned_int(i);
        lcd_print_string("]: ");
        lcd_print_signed_int(w2[i]);

        if (i + 1 < NUM_HIDDEN) {
            lcd_goto_position(2, 1);
            lcd_print_string("w2[");
            lcd_print_unsigned_int(i + 1);
            lcd_print_string("]: ");
            lcd_print_signed_int(w2[i + 1]);
        }

        while (read_keypad() != '#') __delay_ms(50);
        __delay_ms(200);
    }

    // display b2 bias
    lcd_clear_screen();
    lcd_print_string("b2: ");
    lcd_print_signed_int(b2);
    while (read_keypad() != '#') __delay_ms(50);

    lcd_clear_screen();
    lcd_print_string("ready");
}

/* ======== controller ======== */
// main training function called from main.c
// initializes weights, runs training epochs, displays results
void train_network(void) {
    // initialize weights with variety to break symmetry
    // alternating signs and spread values help training converge
    for (uint8_t i = 0; i < NUM_HIDDEN; i++) {
        w1[i] = (i % 2 == 0) ? (400 + i*80) : -(400 + i*80);
        b1[i] = (int16_t)((int16_t)(i - 3) * 100); // -300..200
        w2[i] = (int16_t)(300 + i*30);
    }
    b2 = 128; // small positive bias

    lcd_clear_screen();
    lcd_print_string("training...");

    // run training epochs
    for (int e = 0; e < EPOCHS; e++) {
        train_epoch();
        // show progress every 1000 epochs (takes time on pic)
        if ((e % 1000) == 0) {
            lcd_goto_position(2,1);
            lcd_print_string("epoch:");
            lcd_print_unsigned_int(e);
        }
    }

    lcd_clear_screen();
    lcd_print_string("train done");
    __delay_ms(800);

    // show final weights for copying into run.c
    display_trained_parameters_on_lcd();
}
