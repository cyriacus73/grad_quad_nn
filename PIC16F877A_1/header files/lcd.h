// lcd.h - LCD function declarations
#ifndef LCD_H
#define LCD_H

#include <stdint.h>

void lcd_send_pulse(void);
void lcd_send_command(uint8_t command);
void lcd_send_data(uint8_t data);
void lcd_initialize(void);
void lcd_clear_screen(void);
void lcd_goto_position(uint8_t row, uint8_t col);
void lcd_print_string(const char *text);
void lcd_print_unsigned_int(uint16_t value);
void lcd_print_signed_int(int16_t value);
void lcd_print_fixed_point(int16_t fixed_point_value);

#endif
