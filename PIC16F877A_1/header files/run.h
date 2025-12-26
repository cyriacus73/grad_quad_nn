// run.h
#ifndef RUN_H
#define RUN_H

#include <stdint.h>

// Keypad functions
char read_keypad(void);
int16_t read_decimal_from_keypad(void);

// Neural network inference
int16_t neural_network_predict(int16_t input_x);

// Main inference loop (called from main.c)
void run_network(void);

#endif