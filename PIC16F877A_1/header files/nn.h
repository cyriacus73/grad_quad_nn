#ifndef NN_H
#define NN_H

#include <stdint.h>

/* ================= FIXED-POINT Q8.8 ================= */
#define QMUL(a,b) ((int16_t)(((int32_t)(a) * (b)) >> 8))

/* ================= NETWORK ARCHITECTURE ================= */
#define NN_HIDDEN 6

/* ================= NETWORK PARAMETERS (extern - defined in train.c or run.c) ================= */
extern int16_t w1[NN_HIDDEN];
extern int16_t b1[NN_HIDDEN];
extern int16_t w2[NN_HIDDEN];
extern int16_t b2;

/* ================= ACTIVATION FUNCTIONS ================= */
// Sigmoid: z in Q8.8 ? output in [0, 256]
int16_t sigmoid(int16_t z);

// Sigmoid derivative: h in [0, 256] ? derivative in Q8.8
int16_t sigmoid_deriv(int16_t h);

/* ================= FORWARD PASS ================= */
// x_norm: input in Q8.8 [-256, 256]
// h: buffer for hidden activations (size NN_HIDDEN)
// returns: predicted y in Q8.8 [0, 256]
int16_t nn_forward(int16_t x_norm, int16_t h[NN_HIDDEN]);

#endif /* NN_H */