// lcd.c
#include <xc.h>
#include <stdint.h>
#include "lcd.h"

#define _XTAL_FREQ 20000000

// Control pins
#define RS PORTAbits.RA0
#define RW PORTAbits.RA1
#define EN PORTAbits.RA2

#define LCD_DATA PORTD

void lcd_send_pulse(void) {
    EN = 1;
    __delay_us(1);
    EN = 0;
    __delay_ms(2);
}

void lcd_send_command(uint8_t cmd) {
    RS = 0;
    RW = 0;
    LCD_DATA = cmd;
    lcd_send_pulse();
}

void lcd_send_data(uint8_t data) {
    RS = 1;
    RW = 0;
    LCD_DATA = data;
    lcd_send_pulse();
}


void lcd_initialize(void) {
TRISA = 0x00;
TRISD = 0x00;
    __delay_ms(20);
    lcd_send_command(0x38);  // 8-bit, 2 lines
    lcd_send_command(0x0C);  // display ON
    lcd_send_command(0x01);  // clear display
    __delay_ms(2);
    lcd_send_command(0x06);  // entry mode
}

void lcd_clear_screen(void) {
    lcd_send_command(0x01);
    __delay_ms(2);
}

void lcd_goto_position(uint8_t row, uint8_t col) {
    lcd_send_command((row == 1 ? 0x80 : 0xC0) + (col - 1));
}

void lcd_print_string(const char *text) {
    while (*text) lcd_send_data(*text++);
}

void lcd_print_unsigned_int(uint16_t value)
{
    char buf[6];
    int i = 0;

    if (value == 0) {
        lcd_send_data('0');
        return;
    }

    while (value > 0) {
        buf[i++] = (value % 10) + '0';
        value /= 10;
    }

    while (i--) {
        lcd_send_data(buf[i]);
    }
}

void lcd_print_signed_int(int16_t value)
{
    if (value < 0) {
        lcd_send_data('-');
        value = -value;
    }
    lcd_print_unsigned_int((uint16_t)value);
}

void lcd_print_fixed_point(int16_t q88)
{
    int16_t integer = q88 >> 8;
    uint16_t frac = ((uint16_t)(q88 & 0xFF) * 100) >> 8;

    lcd_print_signed_int(integer);
    lcd_send_data('.');
    lcd_send_data('0' + (frac / 10));
    lcd_send_data('0' + (frac % 10));
}

