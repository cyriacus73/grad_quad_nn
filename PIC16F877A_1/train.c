#include <xc.h>
#include <stdint.h>
#include "train.h"
#include "nn.h"
#include "lcd.h"

#define _XTAL_FREQ 20000000
char read_keypad(void);

/* ================= Fixed-point helpers ================= */
#define QMUL(a,b) ((int16_t)(((int32_t)(a)*(b)) >> 8))
#define CLAMP(x,l,h) ((x)<(l)?(l):((x)>(h)?(h):(x)))

/* ================= Training data ================= */
const int16_t train_x[];
const int16_t train_y[];
#define TRAIN_SAMPLES_COUNT 21

/* ================= Hyperparameters ================= */
#define MAX_EPOCHS  2500
#define LR_Q8_8     ((int16_t)4)   /* 0.015625 */

/* ================= Numeric limits ================= */
#define Z_MIN (-1024)   /* ˜ -4 */
#define Z_MAX (1024)    /* ˜ +4 */

#define W_MIN (-2048)
#define W_MAX (2048)
#define B_MIN (-4096)
#define B_MAX (4096)

/* ================= Network parameters ================= */
int16_t w1[NN_HIDDEN];
int16_t b1[NN_HIDDEN];
int16_t w2[NN_HIDDEN];
int16_t b2;

/* ================= Sigmoid LUT ================= */
/* z ? [-4,+4] mapped to 0..255, output 0..256 */
const int16_t sigmoid_lut_256[256] = {
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0 ,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   1,   1 ,
      1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   2,   2 ,
      2,   2,   2,   2,   2,   2,   3,   3,   3,   3,   3,   3,   4,   4,   4,   4 ,
      5,   5,   5,   6,   6,   6,   7,   7,   8,   8,   9,   9,  10,  10,  11,  12 ,
     12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  24,  25,  26,  28,  29 ,
     31,  33,  35,  37,  39,  41,  43,  45,  48,  50,  53,  55,  58,  61,  64,  67 ,
     70,  73,  77,  80,  84,  87,  91,  95,  98, 102, 106, 110, 114, 118, 122, 126 ,
    130, 134, 138, 142, 146, 150, 154, 158, 161, 165, 169, 172, 176, 179, 183, 186 ,
    189, 192, 195, 198, 201, 203, 206, 208, 211, 213, 215, 217, 219, 221, 223, 225 ,
    227, 228, 230, 231, 232, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244 ,
    244, 245, 246, 246, 247, 247, 248, 248, 249, 249, 250, 250, 250, 251, 251, 251 ,
    252, 252, 252, 252, 253, 253, 253, 253, 253, 253, 254, 254, 254, 254, 254, 254 ,
    254, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
    255, 255, 255, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256 ,
    256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256 ,
};

static inline int16_t sigmoid_q88(int16_t z)
{
    if (z <= Z_MIN) return 0;
    if (z >= Z_MAX) return 256;
    return sigmoid_lut_256[(uint8_t)((z + 1024) >> 3)];
}

static inline int16_t sigmoid_deriv_q88(int16_t h)
{
    return QMUL(h, (int16_t)(256 - h));
}

/* ================= Forward ================= */
int16_t nn_forward(int16_t x, int16_t h[NN_HIDDEN])
{
    int32_t acc = b2;

    for (uint8_t i = 0; i < NN_HIDDEN; ++i) {
        int32_t z = ((int32_t)w1[i] * x) >> 8;
        z += b1[i];
        z = CLAMP(z, Z_MIN, Z_MAX);

        h[i] = sigmoid_q88((int16_t)z);
        acc += ((int32_t)w2[i] * h[i]) >> 8;
    }

    return (int16_t)CLAMP(acc, -32768, 32767);
}

/* ================= Training ================= */
void nn_train_epoch(int16_t lr)
{
    int16_t h[NN_HIDDEN];
    int16_t dh[NN_HIDDEN];

    for (uint8_t s = 0; s < TRAIN_SAMPLES_COUNT; ++s) {

        int16_t y = nn_forward(train_x[s], h);
        int16_t e = y - train_y[s];

        /* ---- output layer ---- */
        b2 = CLAMP(b2 - QMUL(lr, e), B_MIN, B_MAX);

        for (uint8_t i = 0; i < NN_HIDDEN; ++i) {
            int16_t g = QMUL(e, h[i]);
            w2[i] = CLAMP(w2[i] - QMUL(lr, g), W_MIN, W_MAX);
        }

        /* ---- hidden layer ---- */
        for (uint8_t i = 0; i < NN_HIDDEN; ++i) {
            int16_t back = QMUL(e, w2[i]);
            dh[i] = QMUL(back, sigmoid_deriv_q88(h[i]));
        }

        for (uint8_t i = 0; i < NN_HIDDEN; ++i) {
            w1[i] = CLAMP(
                w1[i] - QMUL(lr, QMUL(dh[i], train_x[s])),
                W_MIN, W_MAX
            );
            b1[i] = CLAMP(b1[i] - QMUL(lr, dh[i]), B_MIN, B_MAX);
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
        lcd_print_fixed_point(arr[i+1]);
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

/* ================= Controller ================= */
void train_network(void)
{
    for (uint8_t i = 0; i < NN_HIDDEN; ++i) {
        w1[i] = (i & 1) ? 48 : -64;   /* symmetry broken */
        b1[i] = 0;
        w2[i] = (i & 1) ? 32 : -32;
    }
    b2 = 0;

    for (uint16_t e = 0; e < MAX_EPOCHS; ++e) {
        nn_train_epoch(LR_Q8_8);

        if ((e & 0x1FF) == 0) {
            lcd_clear_screen();
            lcd_print_string("Epoch:");
            lcd_print_unsigned_int(e);
        }
    }

    lcd_clear_screen();
    lcd_print_string("Train Done");
    __delay_ms(400);
    display_trained_parameters_on_lcd();
}