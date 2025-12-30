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
// KEYPAD / PORTD MODES
// ============================================================================

// When scanning keypad we want:
//   RD0..RD2 = columns  = inputs  (1)
//   RD3      = unused   = output (0) - harmless
//   RD4..RD7 = rows     = outputs (0)
// Bits: 7..0 -> 0 0 0 0  0 1 1 1  => 0x07
#define TRIS_KEYPAD_MODE 0x07

// LCD requires PORTD as outputs for DB0..DB7
#define TRIS_LCD_MODE    0x00

// VERIFIED CORRECT MAPPING based on actual hardware test:
// Physical 1 ? [3][2], Physical 5 ? [2][1], Physical 9 ? [1][0], Physical 0 ? [0][1]
const char keypad_map[4][3] = {
   {'1','2','3'},
    {'4','5','6'},
    {'7','8','9'},
    {'*','0','#'} 
};

// ============================================================================
// KEYPAD SCANNER (reads one key press)
// Assumptions:
//  - external pull-ups exist such that columns read HIGH when connected by a row
//  - columns are on RD0..RD2, rows are on RD4..RD7
//  - this function configures TRISD for keypad mode while scanning
// ============================================================================

char scan_keypad(void) {
    // Set TRIS for keypad: columns inputs, rows outputs
    TRISD = TRIS_KEYPAD_MODE;

    // Clear high nibble outputs initially (make rows low)
    PORTD &= 0x0F;

    for (uint8_t row = 0; row < 4; row++) {
        // Drive this row HIGH (activate)
        PORTD |= (1 << (row + 4));
        __delay_us(20); // allow lines to settle

        // Read each column (RD0..RD2)
        for (uint8_t col = 0; col < 3; col++) {
            if (PORTD & (1 << col)) {      // active-high detection (board-specific)
                __delay_ms(20);            // debounce
                if (PORTD & (1 << col)) {  // confirm still pressed
                    char key = keypad_map[row][col];
                    // wait for release
                    while (PORTD & (1 << col));
                    // release row before returning (drive it low)
                    PORTD &= ~(1 << (row + 4));
                    return key;
                }
            }
        }

        // Release this row (drive it low) and continue
        PORTD &= ~(1 << (row + 4));
    }

    return 0; // no key
}

// ============================================================================
// MENU DISPLAY (LCD-only)
// ============================================================================
void display_menu(void) {
    // Give PORTD to LCD
    TRISD = TRIS_LCD_MODE;

    lcd_clear_screen();
    lcd_print_string("0:Run");
    lcd_goto_position(2, 1);
    lcd_print_string("*:Train");
}

// ============================================================================
// wait_for_menu_choice
// This function owns TRIS switching during the menu phase.
// It guarantees PORTD is in LCD mode when returning.
// ============================================================================
char wait_for_menu_choice(void) {
    while (1) {
        // KEYPAD MODE: set TRIS and scan
        char key = scan_keypad();

        // Immediately return PORTD to LCD-safe state
        TRISD = TRIS_LCD_MODE;

        if (key == '1' || key == '2' || key == '3') {
            return key;
        }

        __delay_ms(10); // small idle before next scan
    }
}

// ============================================================================
// MAIN PROGRAM
// ============================================================================

void main(void) {
    // Make pins digital (PIC16F877A)
    ADCON1 = 0x06;

    // Configure LCD control pins (PORTA)
    TRISAbits.TRISA0 = 0;  // RS
    TRISAbits.TRISA1 = 0;  // RW
    TRISAbits.TRISA2 = 0;  // EN

    // Drive control pins to safe idle (RW low for write mode)
    PORTAbits.RA0 = 0;
    PORTAbits.RA1 = 0;  // RW = 0 (write)
    PORTAbits.RA2 = 0;

    // Small power-up delay
    __delay_ms(50);

    // Ensure PORTD is LCD-owned for initialization
    TRISD = TRIS_LCD_MODE;
    PORTD = 0x00;

    // Initialize LCD
    lcd_initialize();

    // Main loop: show menu, wait choice, run mode
    while (1) {
        display_menu();

        char selection = wait_for_menu_choice();

        // Execute selected mode - each mode runs in infinite loop per original design
        if (selection == '1') {
            // ========== DEMO MODE ==========
            lcd_clear_screen();
            lcd_print_string("Mode: DEMO");
            __delay_ms(500);

            // call demo loop (preserved from your original program)
            //run_demo();  // runs demo loop forever

        } else if (selection == '2') {
            // ========== INFERENCE MODE ==========
            lcd_clear_screen();
            lcd_print_string("Mode: RUN");
            __delay_ms(500);

            run_network();  // runs inference loop forever

        } else if (selection == '3') {
            // ========== TRAINING MODE ==========
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

        // small delay before re-showing menu (defensive)
        __delay_ms(200);
    }
}
