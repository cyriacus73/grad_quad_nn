# Neural Network on PIC Microcontroller (Q8.8 Fixed-Point)

## What This Project Does

This project runs a small neural network **entirely on a PIC microcontroller** - no floating-point math, just fixed-point arithmetic. We're teaching it to approximate a quadratic equation (we tested with):

```
y = 3x² - 5x + 7
```

The main point isn't perfect accuracy, but showing that you can train a neural network directly on the microcontroller with these constraints:

* **Fixed-point math only** (Q8.8 format)
* **No floating-point unit**
* **Limited memory and processing power**

The system can:
* Train the network on the PIC itself
* Make predictions on new inputs
* Show results on an LCD screen
* Export the trained weights for later use

---

## Key Design Decisions

### 1. Fixed-Point Arithmetic (Q8.8)

We represent all numbers as signed 16-bit integers:

* Top 8 bits: whole number part
* Bottom 8 bits: decimal part
* Multiply by 256 to get the real value

This gives us predictable math, no weird rounding errors, and fast execution on an 8-bit chip.

No floating-point anywhere in the code.

---

### 2. Input and Output Scaling

The network doesn't work with raw numbers.

* Input `x`: scaled from `[-2, +2]` down to `[-1, +1]`
* Output `y`: scaled from `[4.9, 24]` down to `[0, 1]`

Why bother?

* Keeps neuron outputs in a stable range
* Stops the sigmoid from getting stuck at 0 or 1
* Lets us use smaller weights that train better

Note: The sigmoid's input range (-8 to +8) is separate from this scaling. Different concerns.

---

### 3. Network Architecture

What we built:

* 1 input node
* 6 hidden nodes with sigmoid activation
* 1 output node (linear, no activation)

Kept it small on purpose:

* Fits in the PIC's RAM
* Trains fast enough
* Can still handle a quadratic curve

The network figures out the pattern, not the exact math formula.

---

### 4. Sigmoid Function via Lookup Table

Exp() functions are too slow for a PIC.

Instead:

* 256-entry table that approximates `sigmoid(z)`
* Maps `z` from `[-8, +8]` to table indices `[0, 255]`
* Output scaled to `[0, 256]` in Q8.8

Benefits:

* Instant lookup (no calculation)
* Stable derivative for training
* No complex math at runtime

---

### 5. Training on the Microcontroller

Training uses the basics:
```
* Forward pass
* Backpropagation
* Scaled gradients
* Tuned learning rates
```

With safety measures:

* Error scaling (bit shifts)
* Gradient dampening
* Weight limits
* Smart starting weights

This isn't textbook machine learning - it's embedded ML, adapted for hardware limits.

---

### 6. User Interface

How you interact:

* Keypad for number input
* `*` acts as decimal point
* `#` submits the number
* LCD shows everything

Numbers go straight to fixed-point format.
Example:

* Type `3*4` → means `3.4` → stored as `3.4 × 256`

---

* Forward pass
* Backpropagation
* Manually scaled gradients
* Carefully tuned learning rates

Several safeguards are used:

* Error scaling (right shifts)
* Gradient attenuation
* Weight clamping
* Symmetry-breaking initialization


---

### 7. User Interaction

The system uses:

* A keypad for numeric input
* `*` as a decimal point and (-) sign
* `#` to submit input
* An LCD for all output

Numbers are entered directly into fixed-point format.
For example:

* `3*4` → `3.4` → stored as `3.4 × 256`

---

## Code Files Explained

### `main.c`

**What it does:** System startup and mode selection.

Handles:

* Setting up hardware (ports, LCD, keypad)
* Main menu on screen
* Mode switching:

  * Demo mode
  * Prediction mode
  * Training mode

This file manages the system - no ML code here.

---

### `train.c`

**What it does:** Neural network training.

Handles:

* Training data storage (normalized samples)
* Forward pass
* Backpropagation
* Weight updates
* Training loop with progress
* Exporting final weights

This is where we fought numerical issues:

* Learning rate adjustments
* Gradient scaling
* Avoiding sigmoid saturation
* Preventing weight explosions

All training happens on the PIC.

---

### `run.c`

**What it does:** Making predictions.

Handles:

* Reading keypad input
* Converting to fixed-point
* Input scaling
* Forward pass with trained weights
* Output unscaling
* Showing the result

No training here - just prediction, like in deployment.

---

### `nn.h`

**What it does:** Network setup.

Contains:

* Network size constants
* Shared values
* Fixed-point math helpers

Keeps training and prediction code consistent.

---

### `train.h`

**What it does:** Training interface.

Provides:

* Training function declarations
* Shared parameters between files

Lets `main.c` start training without knowing the details.

---

### `lcd.c / lcd.h`

**What it does:** LCD screen control.

Handles:

* LCD setup
* Cursor movement
* Text and number display

ML code never touches hardware directly.

---

### `demo.c`

**What it does:** Quick testing mode.

Provides:

* Built-in test inputs
* Expected vs actual outputs
* Fast validation without retraining

Great for debugging.


## What We Learned

We ran into lots of numerical headaches:

* Scaling factors being off
* Wrong order of updates
* Sigmoid activations getting stuck at limits
* Input/output scaling not matching up




