#ifndef NN_H
#define NN_H

#include <stdint.h>

// -------- Fixed-point Q8.8 --------
#define QMUL(a,b) ((int16_t)(((int32_t)(a)*(b))>>8))

// -------- Network shape --------
#define NN_INPUTS   1
#define NN_HIDDEN   6
#define NN_OUTPUTS  1

// -------- Parameters --------
// Input -> Hidden
extern int16_t w1[NN_HIDDEN];
extern int16_t b1[NN_HIDDEN];

// Hidden -> Output
extern int16_t w2[NN_HIDDEN];
extern int16_t b2;

// -------- Sigmoid --------
int16_t sigmoid(int16_t x);
int16_t sigmoid_deriv(int16_t x);

// -------- Forward pass --------
int16_t nn_forward(int16_t x, int16_t h[NN_HIDDEN]);

// -------- Training --------
void nn_train_epoch(
    int16_t *x,
    int16_t *y,
    uint8_t n,
    int16_t lr
);

#endif
