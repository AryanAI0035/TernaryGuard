/* Binary UART probe. No heap allocations; all strings live in flash.
 * Request: TG, version=1, mode=1(raw selected) or 2(preprocessed), seq u32,
 *          twenty float32 inputs, CRC16-CCITT (initial 0xffff), little endian.
 * Response: TR, version=1, status, seq u32, pred u8, micros u32, gap u16,
 *           eleven float32 logits, CRC16. Sequence and CRC guard transport.
 */
#include <Arduino.h>
#include <avr/io.h>
#include "nano_infer.h"
static tg_workspace work;
extern char __heap_start;
static uint16_t crc_step(uint16_t crc, uint8_t byte) {
    crc ^= (uint16_t)byte << 8;
    for (uint8_t b=0;b<8;++b) crc = (crc & 0x8000) ? (crc << 1)^0x1021 : crc << 1;
    return crc;
}
static void paint_stack_gap() {
    uint8_t saved=SREG; cli();
    uintptr_t stop=SP-32;
    for (uintptr_t p=(uintptr_t)&__heap_start;p<stop;++p) *(volatile uint8_t*)p=0xA5;
    SREG=saved;
}
static uint16_t untouched_gap() {
    uintptr_t p=(uintptr_t)&__heap_start;
    uintptr_t stop=SP;
    while (p<stop && *(volatile uint8_t*)p==0xA5) ++p;
    return p-(uintptr_t)&__heap_start;
}
static bool read_byte(uint8_t &value) {
    uint32_t start=millis();
    while (!Serial.available()) if ((uint32_t)(millis()-start)>2000) return false;
    value=Serial.read(); return true;
}
static void reply_byte(uint8_t value,uint16_t &crc) {
    Serial.write(value);crc=crc_step(crc,value);
}
void setup() {
    Serial.begin(115200);
    pinMode(8,OUTPUT); // PB0: high only during preprocessing and inference.
    Serial.println(F("TG_NANO_V1 raw_f32=1 preprocessed=2 double_bytes=4"));
    Serial.flush();
    paint_stack_gap();
}
void loop() {
    if (!Serial.available()) return;
    uint8_t h[8];
    if (!read_byte(h[0]) || h[0]!='T' || !read_byte(h[1]) || h[1]!='G') return;
    for (uint8_t i=2;i<8;++i) if (!read_byte(h[i])) return;
    uint16_t crc=0xffff;
    for (uint8_t i=0;i<8;++i) crc=crc_step(crc,h[i]);
    uint8_t *bytes=(uint8_t*)work.a;
    for (uint8_t i=0;i<80;++i) {
        if (!read_byte(bytes[i])) return;
        crc=crc_step(crc,bytes[i]);
    }
    uint8_t low,high;
    if (!read_byte(low) || !read_byte(high)) return;
    uint8_t status=(h[2]!=1 || (h[3]!=1 && h[3]!=2) || crc!=(uint16_t)(low|((uint16_t)high<<8))) ? 1 : 0;
    uint32_t elapsed=0;
    if (!status) {
        PORTB |= _BV(0);
        uint32_t start=micros();
        if ((h[3]==1 && nano_preprocess(work.a)) || tg_infer_preprocessed(&work,work.a,work.b)) status=2;
        elapsed=micros()-start;
        PORTB &= ~_BV(0);
    }
    uint8_t prediction=0;
    for (uint8_t i=1;i<TG_CLASSES;++i) if (work.b[i]>work.b[prediction]) prediction=i;
    uint16_t gap=untouched_gap();
    if (gap<32) status=3;
    crc=0xffff;
    reply_byte('T',crc);reply_byte('R',crc);reply_byte(1,crc);reply_byte(status,crc);
    for (uint8_t i=4;i<8;++i) reply_byte(h[i],crc);
    reply_byte(prediction,crc);
    for (uint8_t i=0;i<4;++i) reply_byte((elapsed>>(8*i))&255,crc);
    reply_byte(gap&255,crc);reply_byte(gap>>8,crc);
    bytes=(uint8_t*)work.b;
    for (uint8_t i=0;i<44;++i) reply_byte(bytes[i],crc);
    Serial.write(crc&255);Serial.write(crc>>8);
}
