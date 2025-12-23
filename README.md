# Grad proj codebase

This project is a PIC16F877A firmware written in C using XC8.
The codebase is split into multiple C and header files.

The system has four main responsibilities:

    System startup and menu control flow
    Neural network training logic (full dataset)
    Interactive demo training (user-provided samples)
    Hardware interaction (LCD and keypad)


main.c:

Configure the microcontroller (I/O ports, pull-ups)
Initialize LCD
Display menu system
Read keypad for mode selection
Route to appropriate mode (Demo, Run, Train)

main.c does NOT:

Contain LCD implementation
Contain keypad implementation details
Contain training logic
Contain neural network math

Typical structure:

Configure hardware
Display menu: "1:Demo 2:Run" / "3:Train"
Wait for user input (1, 2, or 3)
Call appropriate mode function

Three modes:

Demo Mode (press 1): Calls run_demo() from demo.c
Run Mode (press 2): Calls run_network() from run.c
Train Mode (press 3): Calls train_network() and display_trained_parameters_on_lcd() from train.c


lcd.c
lcd.c is the shared LCD driver layer.
This file owns:

All LCD hardware control functions
LCD initialization
LCD character/string output
LCD positioning
LCD number formatting (signed, unsigned, fixed-point)

Examples of what belongs here:

lcd_send_pulse()
lcd_send_command()
lcd_send_data()
lcd_initialize()
lcd_clear_screen()
lcd_goto_position()
lcd_print_string()
lcd_print_signed_int()
lcd_print_unsigned_int()
lcd_print_fixed_point()

lcd.c is the ONLY file that defines these functions.
No other .c file is allowed to define or duplicate them.

lcd.h
lcd.h is the public interface of lcd.c.
It contains:

Function declarations for all LCD functions implemented in lcd.c
Nothing else

It does NOT:

Define functions (only declares them)
Contain logic
Contain global variables with storage

Every file that needs LCD functions includes lcd.h:

main.c
train.c
run.c
demo.c


run.c
run.c is the inference/prediction runtime layer.
This file owns:

Keypad input functions (scanning, debouncing)
Decimal number input from keypad (Q8.8 format)
Trained network parameters (weights and biases)
Neural network forward pass (inference only)
Main inference loop

Examples of what belongs here:

read_keypad()
read_decimal_from_keypad()
calculate_sigmoid()
neural_network_predict()
run_network() - main inference loop

run.c does NOT:

Implement LCD functions (uses lcd.h)
Contain training logic (no backpropagation)
Touch training data

Usage:
Users enter x values via keypad, and the network predicts y values using pre-trained weights.

run.h
run.h is the public interface of run.c.
It contains:

Function declarations for keypad input
Function declaration for run_network()
Function declaration for neural_network_predict()


