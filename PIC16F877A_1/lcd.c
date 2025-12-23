// lcd.c
// Used by all to interface with the lcd


#include "lcd.h"
#include <xc.h>

#define _XTAL_FREQ 20000000

// LCD pin definitions
#define RS RD0
#define RW RD1
#define EN RD2
#define LCD_DATA PORTC

void lcd_send_pulse(void) {
    EN = 1;
    __delay_us(1);
    EN = 0;
    __delay_ms(2);
}

void lcd_send_command(uint8_t command) {//sends commands like clear screen, set cursor, etc
    RS = 0;  // Command mode
    RW = 0;
    LCD_DATA = command;
    lcd_send_pulse();
}

void lcd_send_data(uint8_t data) {//displays just a char on screen
    RS = 1;  // Data mode
    RW = 0;
    LCD_DATA = data;
    lcd_send_pulse();
}

void lcd_initialize(void) {//makes a clean lcd slate
    TRISC = 0x00;
    TRISD = 0x00;
    __delay_ms(20);
    lcd_send_command(0x38);  // 8-bit, 2 lines
    lcd_send_command(0x0C);  // Display on, cursor off
    lcd_send_command(0x01);  // Clear
    __delay_ms(2);
    lcd_send_command(0x06);  // Auto-increment
}

void lcd_clear_screen(void) {
    lcd_send_command(0x01);
    __delay_ms(2);
}

void lcd_goto_position(uint8_t row, uint8_t col) {
    lcd_send_command((row == 1 ? 0x80 : 0xC0) + (col - 1));
}

void lcd_print_string(const char *text) {
    while (*text) {
        lcd_send_data(*text++);
    }
}

void lcd_print_unsigned_int(uint16_t value) {//converts int to ascii cause we can't use stdlib(too heavy for pic)
    char buffer[6];
    int8_t index = 5;
    buffer[index--] = 0;
    
    if (value == 0) {
        buffer[index--] = '0';
    }
    
    while (value) {
        buffer[index--] = '0' + (value % 10);
        value /= 10;
    }
    
    lcd_print_string(&buffer[index + 1]);
}

void lcd_print_signed_int(int16_t value) {
    if (value < 0) {
        lcd_send_data('-');
        value = -value;
    }
    lcd_print_unsigned_int((uint16_t)value);
}

void lcd_print_fixed_point(int16_t fixed_point_value) {//top 8 bits are for integer parts, the bottom 8(in bin)/256 gives the fractionpart 
    // Extract integer part
    int16_t integer_part = fixed_point_value >> 8;
    int16_t fractional_part = (fixed_point_value & 0xFF);
    
    // Handle negative numbers
    if (fixed_point_value < 0) {
        if (fractional_part) {
            integer_part = -(((-fixed_point_value) >> 8) + 1);
            fractional_part = 256 - (fixed_point_value & 0xFF);
        }
    }
    
    lcd_print_signed_int(integer_part);
    lcd_send_data('.');
    
    // Convert fractional part to two decimal digits
    uint16_t fraction_display = ((uint16_t)(fixed_point_value < 0 ? 
        (256 - (fixed_point_value & 0xFF)) : 
        (fixed_point_value & 0xFF)) * 100) >> 8;
    
    if (fraction_display < 10) lcd_send_data('0');
    lcd_print_unsigned_int(fraction_display);
}
