// main.c
// Main entry point - menu system for mode selection
// Shows menu on LCD, user selects mode with keypad

#include <xc.h>
#include <stdint.h>
#include "lcd.h"
#include "train.h"
#include "run.h"
#include "demo.h"


// Configuration bits
#pragma config FOSC = HS, WDTE = OFF, PWRTE = ON, BOREN = ON
#pragma config LVP = OFF, CPD = OFF, WRT = OFF, CP = OFF

#define _XTAL_FREQ 20000000

// ============================================================================
// KEYPAD SCANNING (for menu selection)
// ============================================================================

const char keypad_map[4][3] = {
    {'1', '2', '3'},
    {'4', '5', '6'},
    {'7', '8', '9'},
    {'*', '0', '#'}
};

char get_keypad_input(void) {
    for (uint8_t row = 0; row < 4; row++) {
        PORTB = 0xF0;
        PORTB &= ~(0x10 << row);
        __delay_us(10);
        
        for (uint8_t col = 0; col < 3; col++) {
            if (!(PORTB & (1 << col))) {
                __delay_ms(30);  // Debounce
                while (!(PORTB & (1 << col)));  // Wait for release
                __delay_ms(30);
                return keypad_map[row][col];
            }
        }
    }
    return 0;
}

// ============================================================================
// MENU SYSTEM
// ============================================================================

void display_menu(void) {
    lcd_clear_screen();
    lcd_print_string("1:Demo 2:Run");
    lcd_goto_position(2, 1);
    lcd_print_string("3:Train");
}

char wait_for_menu_choice(void) {
    char key;
    while (1) {
        key = get_keypad_input();
        if (key == '1' || key == '2' || key == '3') {
            return key;
        }
    }
}

// ============================================================================
// MAIN PROGRAM
// ============================================================================

void main(void) {
    // Configure I/O ports
    TRISB = 0x0F;              // Lower 4 bits input (keypad columns)
    TRISC = 0x00;              // PORTC as output (LCD data)
    TRISD = 0x00;              // PORTD as output (LCD control)
    
    // Enable pull-ups on PORTB
    OPTION_REGbits.nRBPU = 0;  // 0 = enable pull-ups
    PORTB = 0xFF;
    
    // Stabilization delay
    __delay_ms(50);
    
    // Initialize LCD
    lcd_initialize();
    lcd_clear_screen();
    
    // Display menu
    display_menu();
    
    // Wait for user selection
    char selection = wait_for_menu_choice();
    
    // Execute selected mode
    if (selection == '1') {
        // ========== DEMO MODE ==========
        lcd_clear_screen();
        lcd_print_string("Mode: DEMO");
        __delay_ms(500);
        
        run_demo();  // Runs forever (loops internally)
        
    } else if (selection == '2') {
        // ========== INFERENCE MODE ==========
        lcd_clear_screen();
        lcd_print_string("Mode: RUN");
        __delay_ms(500);
        
        run_network();  // Runs forever (loops internally)
        
    } else if (selection == '3') {
        // ========== TRAINING MODE ==========
        lcd_clear_screen();
        lcd_print_string("Mode: TRAIN");
        __delay_ms(500);
        
        // Run training (300 epochs)
        train_network();
        
        // Training complete
        lcd_clear_screen();
        lcd_print_string("Train Done");
        __delay_ms(800);
        
        // Display trained parameters (user copies to run.c)
        display_trained_parameters_on_lcd();
        
        // Halt - user views weights
        while (1) {
            __delay_ms(1000);
        }
    }
    
    // Should never reach here (both modes loop forever)
    while (1) {
        __delay_ms(1000);
    }
}