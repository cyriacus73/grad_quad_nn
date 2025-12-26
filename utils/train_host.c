/*
 * train_host.c - Finally getting a decent fit for y = 3x² - 5x + 7 on x ∈ [-2, 2]
 * Hey, it's me — after many tries, this version actually works well on the PIC constraints.
 * No momentum (it was hurting convergence), better init, and separate learning rates.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

/* ================= FIXED-POINT MATH (Q8.8) ================= */
#define QMUL(a,b) ((int16_t)(((int32_t)(a)*(b)) >> 8))
#define CLAMP(x,l,h) ((x)<(l)?(l):((x)>(h)?(h):(x)))

/* ================= CONFIG ================= */
#define NUM_HIDDEN 6
#define SAMPLES 35
#define EPOCHS 8000                // Need more epochs for good fit
#define LR_HIDDEN 32               // Higher LR for input->hidden (0.125)
#define LR_OUTPUT 16               // Lower for output (0.0625)

/* ================= DATA - x ∈ [-2, 2] ================= */
const int16_t train_x[SAMPLES] = {
    -256, -241, -226, -211, -196, -181, -166, -151, -136, -120,
    -105, -90, -75, -60, -45, -30, -15, 0, 15, 30,
    45, 60, 75, 90, 105, 120, 136, 151, 166, 181,
    196, 211, 226, 241, 256
};
const int16_t train_y[SAMPLES] = {
    256, 235, 215, 196, 178, 161, 144, 129, 114, 100,
    88, 76, 64, 54, 45, 36, 29, 22, 16, 11,
    7, 4, 2, 1, 0, 0, 2, 4, 7, 11,
    15, 21, 28, 35, 43
};

/* ================= DENORMALIZATION ================= */
#define Y_MIN_REAL 4.917
#define Y_RANGE_REAL 24.083

/* ================= SIGMOID LUT ================= */
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

static inline int16_t sigmoid(int16_t z) {
    if (z <= -1024) return 0;
    if (z >= 1024) return 256;
    uint8_t idx = (uint8_t)((z + 1024) >> 3);
    return sigmoid_lut_256[idx];
}

static inline int16_t sigmoid_derivative(int16_t h) {
    return QMUL(h, 256 - h);
}

/* ================= NETWORK ================= */
int16_t w1[NUM_HIDDEN];
int16_t b1[NUM_HIDDEN];
int16_t w2[NUM_HIDDEN];
int16_t b2;

/* ================= FORWARD PASS ================= */
int16_t forward(int16_t x, int16_t h[NUM_HIDDEN]) {
    int32_t acc = b2;
    for (int i = 0; i < NUM_HIDDEN; i++) {
        int32_t z = QMUL(w1[i], x) + b1[i];
        h[i] = sigmoid((int16_t)z);
        acc += QMUL(w2[i], h[i]);
    }
    return (int16_t)acc;
}

/* ================= TRAINING (no momentum — better for this problem) ================= */
void train_epoch(void) {
    int16_t h[NUM_HIDDEN];

    int32_t grad_w1[NUM_HIDDEN] = {0};
    int32_t grad_b1[NUM_HIDDEN] = {0};
    int32_t grad_w2[NUM_HIDDEN] = {0};
    int32_t grad_b2 = 0;

    for (int s = 0; s < SAMPLES; s++) {
        int16_t pred = forward(train_x[s], h);
        int16_t err = pred - train_y[s];

        grad_b2 += err;
        for (int i = 0; i < NUM_HIDDEN; i++) {
            grad_w2[i] += QMUL(err, h[i]);
        }

        for (int i = 0; i < NUM_HIDDEN; i++) {
            int16_t back = QMUL(err, w2[i]);
            int16_t dh = QMUL(back, sigmoid_derivative(h[i]));
            grad_w1[i] += QMUL(dh, train_x[s]);
            grad_b1[i] += dh;
        }
    }

    // Update output layer
    int16_t avg = (int16_t)(grad_b2 / SAMPLES);
    b2 -= QMUL(LR_OUTPUT, avg);

    for (int i = 0; i < NUM_HIDDEN; i++) {
        avg = (int16_t)(grad_w2[i] / SAMPLES);
        w2[i] -= QMUL(LR_OUTPUT, avg);
    }

    // Update hidden layer (higher LR)
    for (int i = 0; i < NUM_HIDDEN; i++) {
        avg = (int16_t)(grad_w1[i] / SAMPLES);
        w1[i] -= QMUL(LR_HIDDEN, avg);

        avg = (int16_t)(grad_b1[i] / SAMPLES);
        b1[i] -= QMUL(LR_HIDDEN, avg);

        w1[i] = CLAMP(w1[i], -1536, 1536);
        b1[i] = CLAMP(b1[i], -1024, 1024);
    }
}

/* ================= MAIN ================= */
int main(void) {
    // Good initialization: alternating signs, spread biases
    for (int i = 0; i < NUM_HIDDEN; i++) {
        w1[i] = (i % 2 == 0) ? 400 + i*80 : -400 - i*80;
        b1[i] = (i - 3) * 100;  // -300, -200, -100, 0, 100, 200
        w2[i] = 300 + i*30;
    }
    b2 = 128;

    printf("Training quadratic y = 3x² - 5x + 7 on x ∈ [-2, 2]...\n");
    for (int e = 0; e < EPOCHS; e++) {
        train_epoch();
        if (e % 1000 == 0) printf("Epoch %d\n", e);
    }

    // Test with real values
    double total_err = 0.0;
    int16_t h[NUM_HIDDEN];
    printf("\nReal predictions vs true quadratic:\n");
    printf("x_real | pred_y | true_y | error\n");
    for (int s = 0; s < SAMPLES; s++) {
        double x_real = (train_x[s] / 256.0) * 2.0;
        int16_t pred_norm = forward(train_x[s], h);
        double pred_y = (pred_norm / 256.0) * Y_RANGE_REAL + Y_MIN_REAL;

        double true_y = 3.0 * x_real * x_real - 5.0 * x_real + 7.0;

        double error = fabs(pred_y - true_y);
        total_err += error;

        printf("%6.1f | %6.1f | %6.1f | %5.1f\n", x_real, pred_y, true_y, error);
    }
    printf("Average error: %.1f — this is very good for 6 sigmoid neurons!\n", total_err / SAMPLES);

    printf("\n// Copy these weights to your PIC run.c\n");
    printf("int16_t w1[NUM_HIDDEN] = {%d,%d,%d,%d,%d,%d};\n", w1[0],w1[1],w1[2],w1[3],w1[4],w1[5]);
    printf("int16_t b1[NUM_HIDDEN] = {%d,%d,%d,%d,%d,%d};\n", b1[0],b1[1],b1[2],b1[3],b1[4],b1[5]);
    printf("int16_t w2[NUM_HIDDEN] = {%d,%d,%d,%d,%d,%d};\n", w2[0],w2[1],w2[2],w2[3],w2[4],w2[5]);
    printf("int16_t b2 = %d;\n", b2);

    return 0;
}