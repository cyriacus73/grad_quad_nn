# TinyML Quadratic Regression on PIC16F877A

## Overview

This project demonstrates a tiny machine learning system running entirely on a PIC16F877A microcontroller.
A very small neural model is trained on-device to fit a quadratic relationship between input `x` and output `y`.

The goal is not high accuracy, but to show:

 On-device learning (training on the microcontroller)
 Forward pass, loss computation, and weight updates
 Inference using the trained model
 Feasibility of TinyML under strict memory and compute limits

---

## Model Description

Task: Regression
Model: Quadratic model
  [
  y = ax^2 + bx + c
  ]
Parameters:`a`, `b`, `c` (trained on the MCU)
Training Method: Supervised learning using gradient descent
Loss Function: Mean Squared Error (MSE)

This is equivalent to a 1-neuron neural network with polynomial features.

---

## How Training Works

1. User inputs `(x, y)` data points via keypad.
2. Inputs are normalized to avoid overflow.
3. The model performs a forward pass.
4. Loss is calculated.
5. Weights (`a`, `b`, `c`) are updated using gradient descent.
6. Steps 1–5 repeat for all stored data points.

Training happens entirely on the microcontroller.

---

## How to Use

1. Power the system in Proteus or on hardware.
2. Enter training data points `(x, y)` using the keypad.
3. Start training mode.
4. Wait for training to complete.
5. Enter a new `x` value.
6. The system outputs the predicted `y` on the LCD.

---

## Hardware & Tools

Microcontroller: PIC16F877A
  Compiler:  MPLAB XC8
  Simulation:  Proteus
  Input:  Keypad
  Output:  LCD

---

## Limitations

 Very small dataset (memory constrained)
 Fixed-point arithmetic only
 Low precision
 Simple regression model

These constraints are  intentional  and representative of real TinyML systems.

---

## Educational Purpose

This project demonstrates:

 What TinyML is
 How training differs on embedded systems
 How neural networks can be reduced to simple math
 Tradeoffs between accuracy, memory, and computation

It is designed for learning, not deployment.
