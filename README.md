# Project Summary / README

## What this project is

This project is a PIC16F877A firmware written in C using XC8.
The codebase is split into multiple C files to keep responsibilities clean and to avoid linker conflicts.

The system has three main responsibilities:

1. System startup and control flow
2. Neural network training logic
3. Hardware interaction (LCD and keypad)

Each responsibility lives in exactly one place.

---

## File roles and responsibilities

### main.c

main.c is the entry point of the program.

Its job is minimal by design.

Responsibilities:

* Configure the microcontroller
* Initialize peripherals
* Decide what high-level mode the system runs in
* Call into other modules

main.c does NOT:

* Contain LCD code
* Contain keypad code
* Contain training logic
* Contain neural network math

Typical structure:

* main()
* Infinite loop
* Calls run_network()

---

### run.c

run.c is the hardware interaction and runtime control layer.

This file owns:

* LCD low-level functions
* LCD high-level helper functions
* Keypad input functions
* The main runtime loop that interacts with hardware

Examples of what belongs here:

* lcd_pulse
* lcd_cmd
* lcd_data
* lcd_init
* lcd_clear
* lcd_str
* key_get
* run_network

run.c is the only file that defines these functions.

No other .c file is allowed to define or duplicate them.

---

### run.h

run.h is the public interface of run.c.

It contains:

* Function declarations for functions implemented in run.c
* Nothing else

It does NOT:

* Define functions
* Contain logic
* Contain global variables with storage

Example contents:

* void run_network(void);
* char key_get(void);
* void lcd_init(void);

Every file that wants to use LCD or keypad includes run.h.

---

### train.c

train.c contains the neural network training logic.

Its responsibilities:

* Training data
* Weight updates
* Activation functions
* Cost/error computation
* Any math related to learning

train.c does NOT:

* Touch ports
* Talk to the LCD directly
* Read keypad pins
* Contain LCD or keypad implementations

If train.c needs to display something:

* It calls lcd functions declared in run.h
* It never defines them

---

### train.h

train.h is the public interface of train.c.

It contains:

* Declarations for training-related functions
* Constants related to training

Example:

* void train_step(void);
* void train_init(void);

main.c or run.c can include train.h to trigger training steps.

---

### demo_train.c (important)

demo_train.c was an old or experimental file.

Problem:

* It contained copies of LCD functions
* It was being linked together with run.c

This caused linker errors because:

* The same function names were defined in multiple .c files

Resolution:

* demo_train.c must either be removed from the build
* Or stripped down to pure training logic
* It must not define LCD or keypad functions

---

## Why the linker errors happened

### Function redefined errors

XC8 links all .c files together into one program.

If two files both contain:
void lcd_cmd(...)

The linker sees:

* Two global symbols with the same name
* It does not know which one to use
* Build fails

This is why LCD code must exist in exactly one .c file.

---

### Conflicting declaration errors

This happened because:

* run.c defined run_network()
* run.h declared run_loop()
* main.c called run_network()

The compiler requires:

* The function name
* Return type
* Parameters

to match exactly everywhere.

Even one mismatch causes a hard error.

---

## Design rule we are enforcing

One module, one owner.

* One peripheral → one .c file
* One implementation → one definition
* Other files only see it through a .h file

.c files define behavior
.h files declare behavior

Never the other way around.

---

## Build flow

1. XC8 compiles each .c file separately
2. Each produces an object file
3. The linker combines them
4. Any duplicate symbols cause failure

If it links, the structure is correct.

---

## Current clean architecture

main.c

* includes run.h and train.h
* calls run_network()

run.c

* implements LCD, keypad, runtime loop

run.h

* declares LCD, keypad, runtime functions

train.c

* implements neural network logic

train.h

* declares training functions

No duplicated hardware code anywhere.
