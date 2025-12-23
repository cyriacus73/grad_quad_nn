// demo.h
#ifndef DEMO_H
#define DEMO_H

#include <stdint.h>

// Keypad input
char read_key(void);
int16_t read_number_q88(void);

// Neural network (demo version)
int16_t network_forward(int16_t x, int16_t z[], int16_t h[]);
void train_demo_network(int16_t* x_data, int16_t* y_data, uint8_t num_samples);

// Main demo function
void run_demo(void);

#endif