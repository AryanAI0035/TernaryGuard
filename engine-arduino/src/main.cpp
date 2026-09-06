/*
 * TernaryGuard — Arduino Nano Inference Engine
 * Target: ATmega328P (32KB flash, 2KB RAM)
 * 
 * This file will contain the embedded ternary inference engine.
 * Placeholder — implementation begins in Phase 5.
 * 
 * Protocol (UART, 115200 baud):
 *   Host  → Nano:  feature vector (binary, fixed-point)
 *   Nano  → Host:  classification + latency_us
 */

#include <Arduino.h>

void setup() {
    Serial.begin(115200);
    Serial.println(F("TernaryGuard Arduino Engine — Ready"));
    Serial.println(F("Awaiting model weights (Phase 5)"));
}

void loop() {
    // Phase 5: inference loop will go here
    // 1. Read feature vector from Serial
    // 2. Run ternary inference (add/sub/skip only)
    // 3. Send classification + timing back
    delay(1000);
}
