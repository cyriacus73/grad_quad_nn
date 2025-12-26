// lcd.c
// LCD display interface functions used by all neural network modes
// Handles HD44780 LCD controller communication via 8-bit parallel interface

#include "lcd.h"
#include <xc.h>

#define _XTAL_FREQ 20000000  // 20 MHz crystal for delay timing

// LCD control pin definitions (connected to PORTD)
#define RS RD0  // Register Select: 0=command, 1=data
#define RW RD1  // Read/Write: 0=write, 1=read (always write in this project)
#define EN RD2  // Enable: pulse high to latch data
#define LCD_DATA PORTC  // 8-bit data bus connected to PORTC

// send enable pulse to latch data/command into LCD
void lcd_send_pulse(void) {
    EN = 1;           // enable high
    __delay_us(1);    // minimum pulse width
    EN = 0;           // enable low
    __delay_ms(2);    // wait for LCD to process
}

// send command to LCD (clear screen, set cursor position, etc.)
void lcd_send_command(uint8_t command) {
    RS = 0;              // command mode
    RW = 0;              // write mode
    LCD_DATA = command;  // put command on data bus
    lcd_send_pulse();    // latch command
}

// send character data to LCD (displays character at current cursor position)
void lcd_send_data(uint8_t data) {
    RS = 1;           // data mode
    RW = 0;           // write mode
    LCD_DATA = data;  // put character on data bus
    lcd_send_pulse(); // latch data
}

// initialize LCD in 8-bit mode, 2 lines, cursor settings
void lcd_initialize(void) {
    TRISC = 0x00;  // PORTC output (LCD data bus)
    TRISD = 0x00;  // PORTD output (LCD control signals)

    __delay_ms(20);        // wait for LCD power-up
    lcd_send_command(0x38);  // 8-bit mode, 2 lines, 5x8 font
    lcd_send_command(0x0C);  // display on, cursor off, blink off
    lcd_send_command(0x01);  // clear display
    __delay_ms(2);         // clear command takes longer
    lcd_send_command(0x06);  // increment cursor, no shift
}

// clear LCD screen and return cursor to home position
void lcd_clear_screen(void) {
    lcd_send_command(0x01);  // clear command
    __delay_ms(2);          // wait for clear to complete
}

// move cursor to specified row and column (1-based indexing)
void lcd_goto_position(uint8_t row, uint8_t col) {
    // row 1: 0x80 + (col-1), row 2: 0xC0 + (col-1)
    lcd_send_command((row == 1 ? 0x80 : 0xC0) + (col - 1));
}

// print null-terminated string to LCD at current cursor position
void lcd_print_string(const char *text) {
    while (*text) {           // loop until null terminator
        lcd_send_data(*text++); // send each character
    }
}

// print unsigned integer to LCD (manual conversion, no stdlib)
// converts binary number to decimal ASCII digits
void lcd_print_unsigned_int(uint16_t value) {
    char buffer[6];      // max 5 digits + null for 16-bit
    int8_t index = 5;    // start at end of buffer
    buffer[index--] = 0; // null terminator

    if (value == 0) {
        buffer[index--] = '0';  // handle zero case
    }

    // convert to decimal digits (reverse order)
    while (value) {
        buffer[index--] = '0' + (value % 10);  // get last digit
        value /= 10;                          // remove last digit
    }

    // print from first non-null character
    lcd_print_string(&buffer[index + 1]);
}

// print signed integer to LCD
void lcd_print_signed_int(int16_t value) {
    if (value < 0) {
        lcd_send_data('-');  // negative sign
        value = -value;      // make positive for printing
    }
    lcd_print_unsigned_int((uint16_t)value);  // print magnitude
}

// print q8.8 fixed-point number as decimal (e.g., 1.50, -2.25)
// converts 16-bit fixed-point to LCD display format
void lcd_print_fixed_point(int16_t fixed_point_value) {
    // q8.8 format: bits 15-8 = integer, bits 7-0 = fractional (0-255 represents 0-0.996)

    // extract integer part: shift right by 8
    int16_t integer_part = fixed_point_value >> 8;
    int16_t fractional_part = (fixed_point_value & 0xFF);  // bottom 8 bits

    // handle negative numbers: two's complement adjustment
    if (fixed_point_value < 0) {
        if (fractional_part) {
            // negative with fractional part needs special handling
            integer_part = -(((-fixed_point_value) >> 8) + 1);  // borrow from integer
            fractional_part = 256 - (fixed_point_value & 0xFF);  // complement fractional
        }
    }

    // print integer part
    lcd_print_signed_int(integer_part);
    lcd_send_data('.');  // decimal point

    // convert fractional part to two decimal digits
    // fractional_part * 100 / 256 gives 0-99 for display
    uint16_t fraction_display = ((uint16_t)(fixed_point_value < 0 ?
        (256 - (fixed_point_value & 0xFF)) :  // positive fractional
        (fixed_point_value & 0xFF)) * 100) >> 8;  // scale to 0-99

    if (fraction_display < 10) lcd_send_data('0');  // leading zero
    lcd_print_unsigned_int(fraction_display);       // print decimal digits
}
