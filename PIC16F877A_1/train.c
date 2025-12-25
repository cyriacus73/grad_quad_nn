#include <xc.h>
#include <stdint.h>
#include "train.h"
#include "nn.h"
#include "lcd.h"

#define _XTAL_FREQ 20000000
char read_keypad(void);

/* ================= Fixed-point helpers ================= */
//QMUL multiplies  2 Q8.8 fixed-point numbers and keep the result in Q8.8.
//we cast to int 32 to prevent overflow and shift by 8 bits to convert back to 8.8(cause mult takes it to 16.16)
#define QMUL(a,b) ((int16_t)(((int32_t)(a) * (b)) >> 8)) 

// Clamp a value to a safe range [l, h].
// If x < l ? return l
// If x > h ? return h
// Else ? return x
//prevents weights explosion
#define CLAMP(x,l,h) ((x)<(l)?(l):((x)>(h)?(h):(x)))

/* ================= Training data ================= */
// Generated from Python: x ? [-10, 10], y = 3x² - 5x + 7
const int16_t train_x[35] = {
-256, -241, -226, -211, -196, -181, -166, -151, -136, -120, -105, -90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90, 105, 120, 136, 151, 166, 181, 196, 211, 226, 241, 256
};

const int16_t train_y[35] = {
256, 229, 203, 179, 157, 136, 116, 98, 82, 67, 53, 41, 31, 22, 15, 9, 4, 1, 0, 0, 2, 5, 10, 16, 23, 33, 43, 56, 69, 84, 101, 119, 139, 160, 183
};

#define TRAIN_SAMPLES_COUNT 35

/* ================= Hyperparameters ================= */
#define MAX_EPOCHS  5000
#define LR_Q8_8     ((int16_t)5)   /* 0.01953125 - higher LR since weights are smaller, it's basically LR/256 */

/* ================= Numeric limits ================= */
#define Z_MIN (-1024) 
#define Z_MAX (1024)  //range of Z(for sigmoid) is -4 to 4 [1024/256]

#define W_MIN (-2048)
#define W_MAX (2048) //-8 to 8
#define B_MIN (-4096)
#define B_MAX (4096) //-16 to 16 (doesn't multiply so safe)

/* ================= Network parameters ================= */
int16_t w1[NN_HIDDEN];
int16_t b1[NN_HIDDEN];
int16_t w2[NN_HIDDEN];
int16_t b2;

/* ================= Sigmoid LUT ================= */
const int16_t sigmoid_lut_256[256] = {
      5,   5,   5,   5,   5,   5,   6,   6,   6,   6,   6,   6,   7,   7,   7,   7 ,
      8,   8,   8,   8,   8,   9,   9,   9,  10,  10,  10,  10,  11,  11,  11,  12 ,
     12,  13,  13,  13,  14,  14,  15,  15,  15,  16,  16,  17,  17,  18,  18,  19 ,
     20,  20,  21,  21,  22,  23,  23,  24,  25,  25,  26,  27,  27,  28,  29,  30 ,
     31,  32,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  44,  45,  46 ,
     47,  48,  50,  51,  52,  53,  55,  56,  57,  59,  60,  62,  63,  65,  66,  68 ,
     69,  71,  73,  74,  76,  78,  79,  81,  83,  85,  86,  88,  90,  92,  94,  96 ,
     97,  99, 101, 103, 105, 107, 109, 111, 113, 115, 117, 119, 121, 123, 125, 127 ,
    129, 131, 133, 135, 137, 139, 141, 143, 145, 147, 149, 151, 153, 155, 157, 159 ,
    160, 162, 164, 166, 168, 170, 171, 173, 175, 177, 178, 180, 182, 183, 185, 187 ,
    188, 190, 191, 193, 194, 196, 197, 199, 200, 201, 203, 204, 205, 206, 208, 209 ,
    210, 211, 212, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 224, 225 ,
    226, 227, 228, 229, 229, 230, 231, 231, 232, 233, 233, 234, 235, 235, 236, 236 ,
    237, 238, 238, 239, 239, 240, 240, 241, 241, 241, 242, 242, 243, 243, 243, 244 ,
    244, 245, 245, 245, 246, 246, 246, 246, 247, 247, 247, 248, 248, 248, 248, 248 ,
    249, 249, 249, 249, 250, 250, 250, 250, 250, 250, 251, 251, 251, 251, 251, 251 ,
};

//sigmoid func
//z is from -1024 to 1024 so we shift to make it all positive so (0, 2048)
//bit shift right by 3 so divide by 8 so range is (0, 256)
static inline int16_t sigmoid_q88(int16_t z)
{
    if (z <= Z_MIN) return 0;
    if (z >= Z_MAX) return 256;
    return sigmoid_lut_256[(uint8_t)((z + 1024) >> 3)];
} 

//sigmoid derivative
static inline int16_t sigmoid_deriv_q88(int16_t h)
{
    return QMUL(h, (int16_t)(256 - h));
}

/* ================= Forward ================= */
//hidden layer, h_i ?=s(w1i?x+b1i?)
// y = from i ? ?w2i?hi?+b2? 
int16_t nn_forward(int16_t x, int16_t h[NN_HIDDEN])
{
    int32_t acc = b2;

    for (uint8_t i = 0; i < NN_HIDDEN; ++i) {
        int32_t z = ((int32_t)w1[i] * x) >> 8;
        z += b1[i];
        z = CLAMP(z, Z_MIN, Z_MAX);

        h[i] = sigmoid_q88((int16_t)z);
        acc += ((int32_t)w2[i] * h[i]) >> 8; //formula above
    } 

    return (int16_t)CLAMP(acc, -32768, 32767); //prevents 16bit overflow
}

/* ================= Training ================= */

/* Loss:
L = 0.5 (y_hat - y)^2
dL/dy_hat = e = (y_hat - y) */

void nn_train_epoch(int16_t lr)
{
    int16_t h[NN_HIDDEN];   // hidden activations h_i = s(z_i)
    int16_t dh[NN_HIDDEN];  // hidden deltas d_i = dL/dz_i

    for (uint8_t s = 0; s < TRAIN_SAMPLES_COUNT; ++s) {

        /* ========= FORWARD PASS ========= */
	/* z_i = w1_i * x + b1_i
        h_i = s(z_i)
        y_hat = S (w2_i * h_i) + b2 */
	
        int16_t y = nn_forward(train_x[s], h);

        // Error at output: e = dL/dy_hat = (y_hat - y)
	 int16_t e = y - train_y[s];

        /* ========= BACKWARD: OUTPUT LAYER ========= */

        /* dL/db2 = e
        b2 ? b2 - lr * e     */
        b2 = CLAMP(b2 - QMUL(lr, e), B_MIN, B_MAX);

        /*  dL/dw2_i = e * h_i
        w2_i ? w2_i - lr * (e * h_i)     */
        for (uint8_t i = 0; i < NN_HIDDEN; ++i) {
            int16_t g = QMUL(e, h[i]);          // gradient wrt w2_i
            w2[i] = CLAMP(
                w2[i] - QMUL(lr, g),
                W_MIN, W_MAX
            );
        }


        /* ========= BACKWARD: HIDDEN LAYER ========= */

        for (uint8_t i = 0; i < NN_HIDDEN; ++i) {
            /*
            Backpropagate output error to hidden unit i:
            dL/dh_i = e * w2_i
            */
            int16_t back = QMUL(e, w2[i]);
            /*
            Hidden delta:
            d_i = dL/dz_i
                = (dL/dh_i) * s'(z_i)
                = (e * w2_i) * s'(h_i)
            */
            dh[i] = QMUL(back, sigmoid_deriv_q88(h[i]));
        }


        /* ========= PARAMETER UPDATES: HIDDEN ========= */

        for (uint8_t i = 0; i < NN_HIDDEN; ++i) {

            /*
            dL/dw1_i = d_i * x
            w1_i ? w1_i - lr * d_i * x
            */
            w1[i] = CLAMP(
                w1[i] - QMUL(lr, QMUL(dh[i], train_x[s])),
                W_MIN, W_MAX
            );
            /*
            dL/db1_i = d_i
            b1_i ? b1_i - lr * d_i
            */
            b1[i] = CLAMP(
                b1[i] - QMUL(lr, dh[i]),
                B_MIN, B_MAX
            );
        }
    }
}


/* ================= Display ================= */
static void display_array(const char *name, int16_t *arr)
{
    for (uint8_t i = 0; i < NN_HIDDEN; i += 2) {
        lcd_clear_screen();
        lcd_print_string(name);
        lcd_print_fixed_point(arr[i]);
        lcd_print_string(",");
        if (i+1 < NN_HIDDEN) {
            lcd_print_fixed_point(arr[i+1]);
        }
        while (read_keypad() != '#');
    }
}

void display_trained_parameters_on_lcd(void)
{
    display_array("w1:", w1);
    display_array("b1:", b1);
    display_array("w2:", w2);

    lcd_clear_screen();
    lcd_print_string("b2:");
    lcd_print_fixed_point(b2);
    while (read_keypad() != '#');
}

void train_network(void)
{
    // ========== SMALL WEIGHTS TO PREVENT SATURATION ==========
    // For z = w1*x + b1 to stay in [-1024, 1024]:
    // With x up to 256, we need |w1| << 4.0 (i.e., 1024 in Q8.8)
    // Target: |w1| < 0.5 (128 in Q8.8)
    
    // asymmetric values
    w1[0] = -76;   // -0.297
    w1[1] =  58;   //  0.227
    w1[2] = -45;   // -0.176
    w1[3] =  82;   //  0.320
    w1[4] = -63;   // -0.246
    w1[5] =  49;   //  0.191
    
    b1[0] = -25;   // -0.098
    b1[1] =  18;   //  0.070
    b1[2] =  -8;   // -0.031
    b1[3] =  32;   //  0.125
    b1[4] = -15;   // -0.059
    b1[5] =  21;   //  0.082
    
    w2[0] = -95;   // -0.371
    w2[1] = 112;   //  0.438
    w2[2] = -78;   // -0.305
    w2[3] = 128;   //  0.500
    w2[4] = -88;   // -0.344
    w2[5] = 105;   //  0.410
    
    b2 = 0;

    // ========== TRAINING LOOP ==========
    for (uint16_t e = 0; e < MAX_EPOCHS; ++e) {
        nn_train_epoch(LR_Q8_8);

        // Display progress every 512 epochs
        if ((e & 0x1FF) == 0) {
            lcd_clear_screen();
            lcd_print_string("Epoch:");
            lcd_print_unsigned_int(e);
        }
    }

    lcd_clear_screen();
    lcd_print_string("Train Done");
    __delay_ms(800);
    display_trained_parameters_on_lcd();
}