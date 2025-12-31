#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

/* ================= FIXED-POINT MATH (Q8.8) ================= */
// Q8.8 fixed-point arithmetic: 16-bit signed integers representing real numbers
// Format: bits 15-8 = integer part, bits 7-0 = fractional part (0-255 = 0-0.996)
// Example: 256 = 1.0, -512 = -2.0, 128 = 0.5
// Multiplication: (a*b) >> 8 gives correct q8.8 result
#define QMUL(a,b) ((int16_t)(((int32_t)(a)*(b)) >> 8))
#define CLAMP(x,l,h) ((x)<(l)?(l):((x)>(h)?(h):(x)))

/* ================= CONFIG ================= */
#define NUM_HIDDEN 6            
#define SAMPLES 35                // training samples evenly spaced across x ∈ [-2, 2]
#define EPOCHS 8000              
#define LR_HIDDEN 32             
#define LR_OUTPUT 16              

/* ================= DATA - x ∈ [-2, 2] ================= */
// training data for quadratic function y = 3x² - 5x + 7
// x normalized: x_q88 = x_real * 128 (since x ∈ [-2,2] → [-256,256] in q8.8)
// y normalized: y_q88 = ((y_real - 4.917) / 24.083) * 256
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
// constants for converting normalized neural network output back to real scale
// y_real = y_norm * 24.083 + 4.917 (where y_norm ∈ [0,1])
#define Y_MIN_REAL 4.917    // minimum y value in training data
#define Y_RANGE_REAL 24.083 // range of y values (max - min)

/* ================= SIGMOID LUT ================= */
// lookup table for sigmoid function σ(z) = 1/(1+e^(-z))
// pre-computed for z ∈ [-4, 4] in 256 steps (512 total range, 8 values per step)
// index = (z + 1024) >> 3 maps z ∈ [-1024, 1024] to LUT indices 0-255
// LUT values scaled to q8.8: σ(z) * 256
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

// sigmoid function using LUT: σ(z) = 1/(1+e^(-z))
// clamps extreme values, uses LUT for z ∈ [-4, 4]
static inline int16_t sigmoid(int16_t z) {
    if (z <= -1024) return 0;        // σ(-∞) = 0
    if (z >= 1024) return 256;       // σ(+∞) = 1 (scaled to q8.8)
    uint8_t idx = (uint8_t)((z + 1024) >> 3);  // map z to LUT index
    return sigmoid_lut_256[idx];
}

// derivative of sigmoid: σ'(h) = σ(h) * (1 - σ(h)) = h * (256 - h) in q8.8
// where h = σ(z) is already in q8.8 format [0, 256]
static inline int16_t sigmoid_derivative(int16_t h) {
    return QMUL(h, 256 - h);  // σ' = σ(1-σ), both scaled by 256
}

/* ================= NETWORK ================= */
// neural network parameters: 1 input, 6 hidden sigmoid neurons, 1 linear output
// architecture chosen to approximate quadratic functions with smooth curves
int16_t w1[NUM_HIDDEN];  // input->hidden weights: w1[i] connects input to hidden neuron i
int16_t b1[NUM_HIDDEN];  // hidden biases: b1[i] shifts activation of hidden neuron i
int16_t w2[NUM_HIDDEN];  // hidden->output weights: w2[i] connects hidden neuron i to output
int16_t b2;              // output bias: shifts final output

/* ================= FORWARD PASS ================= */
// implements neural network forward propagation: y = w2 · σ(w1·x + b1) + b2
// mathematical formula: output = Σᵢ w2ᵢ · σ(w1ᵢ·x + b1ᵢ) + b2
// returns prediction in q8.8 format [0, 256] representing normalized y ∈ [0, 1]
int16_t forward(int16_t x, int16_t h[NUM_HIDDEN]) {
    int32_t acc = b2;  // accumulator for output layer (start with bias)

    // compute hidden layer activations and accumulate output
    for (int i = 0; i < NUM_HIDDEN; i++) {
        int32_t z = QMUL(w1[i], x) + b1[i];  // zᵢ = w1ᵢ·x + b1ᵢ
        h[i] = sigmoid((int16_t)z);          // hᵢ = σ(zᵢ)
        acc += QMUL(w2[i], h[i]);            // accumulate w2ᵢ·hᵢ
    }

    return (int16_t)acc;  // final output y
}

/* ================= TRAINING (no momentum — better for this problem) ================= */
// implements one epoch of batch gradient descent with backpropagation
// accumulates gradients over all training samples, then updates weights
// mathematical foundation: ∂L/∂w = (∂L/∂y) * (∂y/∂w) where L = (y_pred - y_true)²/2
void train_epoch(void) {
    int16_t h[NUM_HIDDEN];  // hidden layer activations (computed during forward pass)

    // gradient accumulators (32-bit to prevent overflow during summation)
    int32_t grad_w1[NUM_HIDDEN] = {0};  // ∂L/∂w1ᵢ accumulated over batch
    int32_t grad_b1[NUM_HIDDEN] = {0};  // ∂L/∂b1ᵢ accumulated over batch
    int32_t grad_w2[NUM_HIDDEN] = {0};  // ∂L/∂w2ᵢ accumulated over batch
    int32_t grad_b2 = 0;                // ∂L/∂b2 accumulated over batch

    // forward pass and backward pass for each training sample
    for (int s = 0; s < SAMPLES; s++) {
        int16_t pred = forward(train_x[s], h);     // forward: get prediction
        int16_t err = pred - train_y[s];           // error: ε = y_pred - y_true

        // output layer gradients (simpler: no activation derivative)
        grad_b2 += err;                            // ∂L/∂b2 = ε
        for (int i = 0; i < NUM_HIDDEN; i++) {
            grad_w2[i] += QMUL(err, h[i]);         // ∂L/∂w2ᵢ = ε · hᵢ
        }

        // hidden layer gradients (chain rule through sigmoid)
        for (int i = 0; i < NUM_HIDDEN; i++) {
            int16_t back = QMUL(err, w2[i]);       // backpropagated error: δᵢ = ε · w2ᵢ
            int16_t dh = QMUL(back, sigmoid_derivative(h[i]));  // δᵢ *= σ'(hᵢ)
            grad_w1[i] += QMUL(dh, train_x[s]);    // ∂L/∂w1ᵢ = δᵢ · x
            grad_b1[i] += dh;                       // ∂L/∂b1ᵢ = δᵢ
        }
    }

    // update output layer weights (lower learning rate for stability)
    int16_t avg = (int16_t)(grad_b2 / SAMPLES);    // average gradient over batch
    b2 -= QMUL(LR_OUTPUT, avg);                    // b2 -= η · ∂L/∂b2

    for (int i = 0; i < NUM_HIDDEN; i++) {
        avg = (int16_t)(grad_w2[i] / SAMPLES);
        w2[i] -= QMUL(LR_OUTPUT, avg);             // w2ᵢ -= η · ∂L/∂w2ᵢ
    }

    // update hidden layer weights (higher learning rate for faster convergence)
    for (int i = 0; i < NUM_HIDDEN; i++) {
        avg = (int16_t)(grad_w1[i] / SAMPLES);
        w1[i] -= QMUL(LR_HIDDEN, avg);             // w1ᵢ -= η · ∂L/∂w1ᵢ

        avg = (int16_t)(grad_b1[i] / SAMPLES);
        b1[i] -= QMUL(LR_HIDDEN, avg);             // b1ᵢ -= η · ∂L/∂b1ᵢ

        // clamp weights to prevent overflow in q8.8 arithmetic
        w1[i] = CLAMP(w1[i], -1536, 1536);         // ±6.0 in q8.8
        b1[i] = CLAMP(b1[i], -1024, 1024);         // ±4.0 in q8.8
    }
}

/* ================= MAIN ================= */
// host training program: trains neural network on PC, exports weights for PIC
// this program replicates the PIC training algorithm to verify correctness
int main(void) {
    // initialize weights with alternating signs and spread biases
    // good initialization prevents symmetry and helps convergence
    for (int i = 0; i < NUM_HIDDEN; i++) {
        w1[i] = (i % 2 == 0) ? 400 + i*80 : -400 - i*80;  // alternating ±4.0 to ±6.0
        b1[i] = (i - 3) * 100;  // -300, -200, -100, 0, 100, 200 (spread biases)
        w2[i] = 300 + i*30;     // 3.0 to 4.8 (positive output weights)
    }
    b2 = 128;  // 0.5 bias for output layer

    // train the network on quadratic function y = 3x² - 5x + 7
    printf("Training quadratic y = 3x² - 5x + 7 on x ∈ [-2, 2]...\n");
    for (int e = 0; e < EPOCHS; e++) {
        train_epoch();  // one epoch of batch gradient descent
       
    }

    // print predicted curve as Python arrays
    printf("\nQuadratic curve y = 3x^2 - 5x + 7 as Python arrays:\n");
    printf("x_vals = [");
    for (int s = 0; s < SAMPLES; s++) {
        double x = (train_x[s] / 256.0) * 2.0;
        printf("%.1f", x);
        if (s < SAMPLES-1) printf(", ");
    }
    printf("]\n");

    int16_t h_pred[NUM_HIDDEN];
    printf("y_vals = [");
    for (int s = 0; s < SAMPLES; s++) {
        int16_t pred_norm = forward(train_x[s], h_pred);
        double pred_y = (pred_norm / 256.0) * Y_RANGE_REAL + Y_MIN_REAL;
        printf("%.1f", pred_y);
        if (s < SAMPLES-1) printf(", ");
    }
    printf("]\n");

    // test the trained network on all training samples
    double total_err = 0.0;
    int16_t h[NUM_HIDDEN];
    printf("\nReal predictions vs true quadratic:\n");
    printf("x_real | pred_y | true_y | error\n");
    for (int s = 0; s < SAMPLES; s++) {
        // convert normalized inputs back to real scale for display
        double x_real = (train_x[s] / 256.0) * 2.0;  // q8.8 to real: x_real = x_q88 / 128

        // forward pass to get normalized prediction
        int16_t pred_norm = forward(train_x[s], h);

        // denormalize prediction: y_real = y_norm * range + min
        double pred_y = (pred_norm / 256.0) * Y_RANGE_REAL + Y_MIN_REAL;

        // compute true quadratic value
        double true_y = 3.0 * x_real * x_real - 5.0 * x_real + 7.0;

        // calculate and accumulate error
        double error = fabs(pred_y - true_y);
        total_err += error;

        printf("%6.1f | %6.1f | %6.1f | %5.1f\n", x_real, pred_y, true_y, error);
    }

    printf("Average error: %.1f \n", total_err / SAMPLES);

    // export trained weights for copying to PIC run.c
    printf("\n// weigths to test on PIC run.c\n");
    printf("int16_t w1[NUM_HIDDEN] = {%d,%d,%d,%d,%d,%d};\n", w1[0],w1[1],w1[2],w1[3],w1[4],w1[5]);
    printf("int16_t b1[NUM_HIDDEN] = {%d,%d,%d,%d,%d,%d};\n", b1[0],b1[1],b1[2],b1[3],b1[4],b1[5]);
    printf("int16_t w2[NUM_HIDDEN] = {%d,%d,%d,%d,%d,%d};\n", w2[0],w2[1],w2[2],w2[3],w2[4],w2[5]);
    printf("int16_t b2 = %d;\n", b2);

    return 0;
}