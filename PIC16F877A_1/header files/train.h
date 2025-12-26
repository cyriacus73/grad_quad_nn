#ifndef TRAIN_H
#define TRAIN_H

#include <stdint.h>

#define NN_HIDDEN 6

// Declarations matching the new implementation
void train_network(void);
void display_trained_parameters_on_lcd(void);
void export_weights_to_pc(void);
void nn_train_epoch(int16_t lr); // Removed extra pointers to match train.c

// Make variables visible to other files
extern int16_t w1[NN_HIDDEN];
extern int16_t b1[NN_HIDDEN];
extern int16_t w2[NN_HIDDEN];
extern int16_t b2;

// The 256-entry table
extern const int16_t sigmoid_lut_256[256];

#endif