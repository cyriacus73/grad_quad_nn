// main.c
// Main entry point - menu system for mode selection
// Shows menu on LCD, user selects mode with keypad
// This is the PIC microcontroller's main program that runs when powered on

#include <xc.h>
#include <stdint.h>
#include "lcd.h"
#include "train.h"
#include "run.h"
#include "demo.h"


// Configuration bits - tell the PIC how to start up
#pragma config FOSC = HS, WDTE = OFF, PWRTE = ON, BOREN = ON
#pragma config LVP = OFF, CPD = OFF, WRT = OFF, CP = OFF

#define _XTAL_FREQ 20000000  // 20 MHz crystal oscillator

// ============================================================================
// KEYPAD SCANNING (for menu selection)
// ============================================================================

// keypad layout: 4 rows x 3 columns connected to PORTB
const char keypad_map[4][3] = {
    {'1', '2', '3'},
    {'4', '5', '6'},
    {'7', '8', '9'},
    {'*', '0', '#'}
};

// matrix keypad scanner - scans rows/columns to detect key presses
// returns the character of pressed key or 0 if none
char get_keypad_input(void) {
    // scan each row by setting it low and checking columns
    for (uint8_t row = 0; row < 4; row++) {
        PORTB = 0xF0;                    // set all rows high (RB4-RB7 = 1)
        PORTB &= ~(0x10 << row);         // pull current row low (RB4+row = 0)
        __delay_us(10);                  // settling delay

        // check each column for key press (low signal)
        for (uint8_t col = 0; col < 3; col++) {
            if (!(PORTB & (1 << col))) { // column RB0-RB2 is low = key pressed
                __delay_ms(30);          // debounce delay
                while (!(PORTB & (1 << col))); // wait for key release
                __delay_ms(30);          // additional debounce
                return keypad_map[row][col];
            }
        }
    }
    return 0;  // no key pressed
}

// ============================================================================
// MENU SYSTEM
// ============================================================================

// display the main menu on LCD showing available modes
void display_menu(void) {
    lcd_clear_screen();
    lcd_print_string("1:Demo 2:Run");
    lcd_goto_position(2, 1);
    lcd_print_string("3:Train");
}

// wait for user to press 1, 2, or 3 on keypad
// returns the selected menu option
char wait_for_menu_choice(void) {
    char key;
    while (1) {
        key = get_keypad_input();
        if (key == '1' || key == '2' || key == '3') {
            return key;
        }
        // ignore other keys, keep waiting
    }
}

// ============================================================================
// MAIN PROGRAM
// ============================================================================

// main entry point - runs when PIC powers on or resets
// initializes hardware, shows menu, runs selected neural network mode
void main(void) {
    // configure I/O ports for keypad and LCD
    TRISB = 0x0F;              // RB0-RB3 input (keypad columns), RB4-RB7 output (keypad rows)
    TRISC = 0x00;              // PORTC output (LCD data bus)
    TRISD = 0x00;              // PORTD output (LCD control signals)

    // enable internal pull-up resistors on PORTB for keypad
    OPTION_REGbits.nRBPU = 0;  // 0 = enable pull-ups
    PORTB = 0xFF;              // set PORTB high

    // wait for power supply to stabilize
    __delay_ms(50);

    // initialize LCD display
    lcd_initialize();
    lcd_clear_screen();

    // show main menu to user
    display_menu();

    // wait for user to select mode (1=Demo, 2=Run, 3=Train)
    char selection = wait_for_menu_choice();

    // execute selected mode - each mode runs in infinite loop
    if (selection == '1') {
        // ========== DEMO MODE ==========
        // shows pre-recorded neural network predictions
        lcd_clear_screen();
        lcd_print_string("Mode: DEMO");
        __delay_ms(500);

        run_demo();  // runs demo loop forever

    } else if (selection == '2') {
        // ========== INFERENCE MODE ==========
        // user inputs x, neural network predicts y
        lcd_clear_screen();
        lcd_print_string("Mode: RUN");
        __delay_ms(500);

        run_network();  // runs inference loop forever

    } else if (selection == '3') {
        // ========== TRAINING MODE ==========
        // trains neural network on PIC, displays learned weights
        lcd_clear_screen();
        lcd_print_string("Mode: TRAIN");
        __delay_ms(500);

        // run training algorithm (300 epochs of gradient descent)
        train_network();

        // training complete - show completion message
        lcd_clear_screen();
        lcd_print_string("Train Done");
        __delay_ms(800);

        // display trained weights on LCD (user copies to run.c)
        display_trained_parameters_on_lcd();

        // halt here - user views the weights for manual copying
        while (1) {
            __delay_ms(1000);
        }
    }

    // should never reach here - all modes loop forever
    while (1) {
        __delay_ms(1000);
    }
}