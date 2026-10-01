# Phase 5 progress — NOT hardware validated

The AVR port builds, but no Nano USB serial device is connected/visible. No flash
attempt, physical serial validation, board latency measurement or SRAM peak
measurement has been performed. Phase 5 is incomplete; Phase 6 is not started.

Frozen artifacts and Phase 4 engine are unchanged. Preprocessing is implemented
on-device in float32 as a candidate, pending AVR-libc/hardware acceptance.
The host build of the AVR source gives zero prediction disagreements on 69,040
source-verified frozen test rows and TCP 1/5,555; maximum preprocessing difference
is 1.430511474609375e-06 and maximum logit difference is 2.288818359375e-05 versus
the Phase 4 C reference. The Phase 4 reference itself is checked against PyTorch
on all rows in the same preparation command. These are host results only.

An earlier exploratory ctypes probe passed strided input rows directly to C and
reported invalid predictions. It was discarded, corrected with contiguous arrays,
and replaced by the checked-in preparation script, which asserts row layout.
No model edits were made in response to the invalid probe.

AVR-GCC 7.3.0 final ELF: 8,076 flash bytes and 1,212 static SRAM bytes. Neither
number is an actual runtime SRAM peak. Model, bias, gamma, scale and preprocessing
constants reside in .text/PROGMEM; there are no linked heap allocator symbols.
The actual linked AVR packed-dot function and integer helpers contain no multiply
instructions or floating-point helper calls. The toolchain build asserts that
AVR double and float are both 4 bytes.

Proposed hardware plan (pending agreement): 330 stratified samples, 30 per class,
including the sole correct TCP row at test index 59782, plus all 5,555 TCP rows
in a separate pass. This does not imply full 69,040-row hardware coverage.
Known limitation must remain exactly 1/5,555 (0.018%) on that complete TCP pass.

99 tests pass (96 before the three new Nano host/protocol tests). These tests
are not physical board measurements. See engine-arduino/README.md for build,
flash, serial protocol, measurement limitations and resume commands.

## Connected ports

```text
$ /private/tmp/ternaryguard-phase5-tools/bin/python -m serial.tools.list_ports -v
/dev/cu.Bluetooth-Incoming-Port
    desc: n/a
    hwid: n/a
/dev/cu.debug-console
    desc: n/a
    hwid: n/a
2 ports found

```

## AVR build

```text
$ PLATFORMIO_CORE_DIR=/private/tmp/ternaryguard-phase5-pio /private/tmp/ternaryguard-phase5-tools/bin/pio run -d engine-arduino -e nano_new
Processing nano_new (board: nanoatmega328new; platform: atmelavr@5.3.0; framework: arduino)
--------------------------------------------------------------------------------
Verbose mode can be enabled via `-v, --verbose` option
CONFIGURATION: https://docs.platformio.org/page/boards/atmelavr/nanoatmega328new.html
PLATFORM: Atmel AVR (5.3.0) > Arduino Nano ATmega328 (New Bootloader)
HARDWARE: ATMEGA328P 16MHz, 2KB RAM, 30KB Flash
DEBUG: Current (avr-stub) External (avr-stub, simavr)
PACKAGES: 
 - framework-arduino-avr @ 5.4.0 
 - toolchain-atmelavr @ 1.70300.191015 (7.3.0)
LDF: Library Dependency Finder -> https://bit.ly/configure-pio-ldf
LDF Modes: Finder ~ chain, Compatibility ~ soft
Found 5 compatible libraries
Scanning dependencies...
No dependencies
Building in release mode
Compiling .pio/build/nano_new/src/main.cpp.o
Compiling .pio/build/nano_new/src/ternary_avr.c.o
Archiving .pio/build/nano_new/libFrameworkArduinoVariant.a
Compiling .pio/build/nano_new/FrameworkArduino/CDC.cpp.o
Compiling .pio/build/nano_new/FrameworkArduino/HardwareSerial.cpp.o
Compiling .pio/build/nano_new/FrameworkArduino/HardwareSerial0.cpp.o
Compiling .pio/build/nano_new/FrameworkArduino/HardwareSerial1.cpp.o
Compiling .pio/build/nano_new/FrameworkArduino/HardwareSerial2.cpp.o
Indexing .pio/build/nano_new/libFrameworkArduinoVariant.a
Compiling .pio/build/nano_new/FrameworkArduino/HardwareSerial3.cpp.o
Compiling .pio/build/nano_new/FrameworkArduino/IPAddress.cpp.o
Compiling .pio/build/nano_new/FrameworkArduino/PluggableUSB.cpp.o
Compiling .pio/build/nano_new/FrameworkArduino/Print.cpp.o
Compiling .pio/build/nano_new/FrameworkArduino/Stream.cpp.o
Compiling .pio/build/nano_new/FrameworkArduino/Tone.cpp.o
Compiling .pio/build/nano_new/FrameworkArduino/USBCore.cpp.o
Compiling .pio/build/nano_new/FrameworkArduino/WInterrupts.c.o
Compiling .pio/build/nano_new/FrameworkArduino/WMath.cpp.o
Compiling .pio/build/nano_new/FrameworkArduino/WString.cpp.o
Compiling .pio/build/nano_new/FrameworkArduino/abi.cpp.o
Compiling .pio/build/nano_new/FrameworkArduino/hooks.c.o
Compiling .pio/build/nano_new/FrameworkArduino/main.cpp.o
Compiling .pio/build/nano_new/FrameworkArduino/new.cpp.o
Compiling .pio/build/nano_new/FrameworkArduino/wiring.c.o
Compiling .pio/build/nano_new/FrameworkArduino/wiring_analog.c.o
Compiling .pio/build/nano_new/FrameworkArduino/wiring_digital.c.o
/private/tmp/ternaryguard-phase5-pio/packages/framework-arduino-avr/cores/arduino/new.cpp: In function 'void* operator new(std::size_t, std::nothrow_t)':
/private/tmp/ternaryguard-phase5-pio/packages/framework-arduino-avr/cores/arduino/new.cpp:59:60: warning: unused parameter 'tag' [-Wunused-parameter]
 void * operator new(std::size_t size, const std::nothrow_t tag) noexcept {
                                                            ^~~
/private/tmp/ternaryguard-phase5-pio/packages/framework-arduino-avr/cores/arduino/new.cpp: In function 'void* operator new [](std::size_t, const std::nothrow_t&)':
/private/tmp/ternaryguard-phase5-pio/packages/framework-arduino-avr/cores/arduino/new.cpp:68:63: warning: unused parameter 'tag' [-Wunused-parameter]
 void * operator new[](std::size_t size, const std::nothrow_t& tag) noexcept {
                                                               ^~~
/private/tmp/ternaryguard-phase5-pio/packages/framework-arduino-avr/cores/arduino/new.cpp: In function 'void operator delete(void*, const std::nothrow_t&)':
/private/tmp/ternaryguard-phase5-pio/packages/framework-arduino-avr/cores/arduino/new.cpp:103:55: warning: unused parameter 'tag' [-Wunused-parameter]
 void operator delete(void* ptr, const std::nothrow_t& tag) noexcept {
                                                       ^~~
/private/tmp/ternaryguard-phase5-pio/packages/framework-arduino-avr/cores/arduino/new.cpp: In function 'void operator delete [](void*, const std::nothrow_t&)':
/private/tmp/ternaryguard-phase5-pio/packages/framework-arduino-avr/cores/arduino/new.cpp:106:57: warning: unused parameter 'tag' [-Wunused-parameter]
 void operator delete[](void* ptr, const std::nothrow_t& tag) noexcept {
                                                         ^~~
Compiling .pio/build/nano_new/FrameworkArduino/wiring_pulse.S.o
Compiling .pio/build/nano_new/FrameworkArduino/wiring_pulse.c.o
Compiling .pio/build/nano_new/FrameworkArduino/wiring_shift.c.o
Archiving .pio/build/nano_new/libFrameworkArduino.a
Indexing .pio/build/nano_new/libFrameworkArduino.a
Linking .pio/build/nano_new/firmware.elf
Checking size .pio/build/nano_new/firmware.elf
Advanced Memory Usage is available via "PlatformIO Home > Project Inspect"
RAM:   [======    ]  59.2% (used 1212 bytes from 2048 bytes)
Flash: [===       ]  26.3% (used 8076 bytes from 30720 bytes)
Building .pio/build/nano_new/firmware.hex
========================= [SUCCESS] Took 0.97 seconds =========================

Environment    Status    Duration
-------------  --------  ------------
nano_new       SUCCESS   00:00:00.971
========================= 1 succeeded in 00:00:00.971 =========================

```

## Size, linked AVR kernel and flash placement

```text
$ Commands and extracted linked disassembly below
$ /private/tmp/ternaryguard-phase5-pio/packages/toolchain-atmelavr/bin/avr-size -C --mcu=atmega328p engine-arduino/.pio/build/nano_new/firmware.elf
AVR Memory Usage
----------------
Device: atmega328p

Program:    8076 bytes (24.6% Full)
(.text + .data + .bootloader)

Data:       1212 bytes (59.2% Full)
(.data + .bss + .noinit)




$ /private/tmp/ternaryguard-phase5-pio/packages/toolchain-atmelavr/bin/avr-size -A engine-arduino/.pio/build/nano_new/firmware.elf
engine-arduino/.pio/build/nano_new/firmware.elf  :
section                     size      addr
.data                         22   8388864
.text                       8054         0
.bss                        1190   8388886
.comment                      17         0
.note.gnu.avr.deviceinfo      64         0
.debug_aranges               224         0
.debug_info                 2742         0
.debug_abbrev               1582         0
.debug_line                  886         0
.debug_str                   520         0
Total                      15301




$ /private/tmp/ternaryguard-phase5-pio/packages/toolchain-atmelavr/bin/avr-gcc --version
avr-gcc (GCC) 7.3.0
Copyright (C) 2017 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.



Linked AVR packed-dot kernel:
00000d34 <tg_packed_dot>:
     d34:	2f 92       	push	r2
     d36:	3f 92       	push	r3
     d38:	4f 92       	push	r4
     d3a:	5f 92       	push	r5
     d3c:	6f 92       	push	r6
     d3e:	7f 92       	push	r7
     d40:	8f 92       	push	r8
     d42:	9f 92       	push	r9
     d44:	af 92       	push	r10
     d46:	bf 92       	push	r11
     d48:	cf 92       	push	r12
     d4a:	df 92       	push	r13
     d4c:	ef 92       	push	r14
     d4e:	ff 92       	push	r15
     d50:	0f 93       	push	r16
     d52:	1f 93       	push	r17
     d54:	cf 93       	push	r28
     d56:	df 93       	push	r29
     d58:	2c 01       	movw	r4, r24
     d5a:	3b 01       	movw	r6, r22
     d5c:	19 01       	movw	r2, r18
     d5e:	e8 01       	movw	r28, r16
     d60:	da 01       	movw	r26, r20
     d62:	91 2c       	mov	r9, r1
     d64:	81 2c       	mov	r8, r1
     d66:	20 e0       	ldi	r18, 0x00	; 0
     d68:	30 e0       	ldi	r19, 0x00	; 0
     d6a:	40 e0       	ldi	r20, 0x00	; 0
     d6c:	50 e0       	ldi	r21, 0x00	; 0
     d6e:	60 e0       	ldi	r22, 0x00	; 0
     d70:	70 e0       	ldi	r23, 0x00	; 0
     d72:	80 e0       	ldi	r24, 0x00	; 0
     d74:	90 e0       	ldi	r25, 0x00	; 0
     d76:	83 01       	movw	r16, r6
     d78:	08 0d       	add	r16, r8
     d7a:	19 1d       	adc	r17, r9
     d7c:	f8 01       	movw	r30, r16
     d7e:	f6 95       	lsr	r31
     d80:	e7 95       	ror	r30
     d82:	f6 95       	lsr	r31
     d84:	e7 95       	ror	r30
     d86:	e4 0d       	add	r30, r4
     d88:	f5 1d       	adc	r31, r5
     d8a:	e4 91       	lpm	r30, Z
     d8c:	f0 e0       	ldi	r31, 0x00	; 0
     d8e:	00 0f       	add	r16, r16
     d90:	11 1f       	adc	r17, r17
     d92:	06 70       	andi	r16, 0x06	; 6
     d94:	11 27       	eor	r17, r17
     d96:	02 c0       	rjmp	.+4      	; 0xd9c <tg_packed_dot+0x68>
     d98:	f5 95       	asr	r31
     d9a:	e7 95       	ror	r30
     d9c:	0a 95       	dec	r16
     d9e:	e2 f7       	brpl	.-8      	; 0xd98 <tg_packed_dot+0x64>
     da0:	e3 70       	andi	r30, 0x03	; 3
     da2:	e1 30       	cpi	r30, 0x01	; 1
     da4:	e1 f5       	brne	.+120    	; 0xe1e <tg_packed_dot+0xea>
     da6:	ac 90       	ld	r10, X
     da8:	11 96       	adiw	r26, 0x01	; 1
     daa:	bc 90       	ld	r11, X
     dac:	11 97       	sbiw	r26, 0x01	; 1
     dae:	12 96       	adiw	r26, 0x02	; 2
     db0:	cc 90       	ld	r12, X
     db2:	12 97       	sbiw	r26, 0x02	; 2
     db4:	13 96       	adiw	r26, 0x03	; 3
     db6:	dc 90       	ld	r13, X
     db8:	13 97       	sbiw	r26, 0x03	; 3
     dba:	14 96       	adiw	r26, 0x04	; 4
     dbc:	ec 90       	ld	r14, X
     dbe:	14 97       	sbiw	r26, 0x04	; 4
     dc0:	15 96       	adiw	r26, 0x05	; 5
     dc2:	fc 90       	ld	r15, X
     dc4:	15 97       	sbiw	r26, 0x05	; 5
     dc6:	16 96       	adiw	r26, 0x06	; 6
     dc8:	0c 91       	ld	r16, X
     dca:	16 97       	sbiw	r26, 0x06	; 6
     dcc:	17 96       	adiw	r26, 0x07	; 7
     dce:	1c 91       	ld	r17, X
     dd0:	17 97       	sbiw	r26, 0x07	; 7
     dd2:	0e 94 cc 0b 	call	0x1798	; 0x1798 <__adddi3>
     dd6:	ef ef       	ldi	r30, 0xFF	; 255
     dd8:	8e 1a       	sub	r8, r30
     dda:	9e 0a       	sbc	r9, r30
     ddc:	18 96       	adiw	r26, 0x08	; 8
     dde:	28 14       	cp	r2, r8
     de0:	39 04       	cpc	r3, r9
     de2:	49 f6       	brne	.-110    	; 0xd76 <tg_packed_dot+0x42>
     de4:	28 83       	st	Y, r18
     de6:	39 83       	std	Y+1, r19	; 0x01
     de8:	4a 83       	std	Y+2, r20	; 0x02
     dea:	5b 83       	std	Y+3, r21	; 0x03
     dec:	6c 83       	std	Y+4, r22	; 0x04
     dee:	7d 83       	std	Y+5, r23	; 0x05
     df0:	8e 83       	std	Y+6, r24	; 0x06
     df2:	9f 83       	std	Y+7, r25	; 0x07
     df4:	90 e0       	ldi	r25, 0x00	; 0
     df6:	80 e0       	ldi	r24, 0x00	; 0
     df8:	df 91       	pop	r29
     dfa:	cf 91       	pop	r28
     dfc:	1f 91       	pop	r17
     dfe:	0f 91       	pop	r16
     e00:	ff 90       	pop	r15
     e02:	ef 90       	pop	r14
     e04:	df 90       	pop	r13
     e06:	cf 90       	pop	r12
     e08:	bf 90       	pop	r11
     e0a:	af 90       	pop	r10
     e0c:	9f 90       	pop	r9
     e0e:	8f 90       	pop	r8
     e10:	7f 90       	pop	r7
     e12:	6f 90       	pop	r6
     e14:	5f 90       	pop	r5
     e16:	4f 90       	pop	r4
     e18:	3f 90       	pop	r3
     e1a:	2f 90       	pop	r2
     e1c:	08 95       	ret
     e1e:	e2 30       	cpi	r30, 0x02	; 2
     e20:	c9 f4       	brne	.+50     	; 0xe54 <tg_packed_dot+0x120>
     e22:	ac 90       	ld	r10, X
     e24:	11 96       	adiw	r26, 0x01	; 1
     e26:	bc 90       	ld	r11, X
     e28:	11 97       	sbiw	r26, 0x01	; 1
     e2a:	12 96       	adiw	r26, 0x02	; 2
     e2c:	cc 90       	ld	r12, X
     e2e:	12 97       	sbiw	r26, 0x02	; 2
     e30:	13 96       	adiw	r26, 0x03	; 3
     e32:	dc 90       	ld	r13, X
     e34:	13 97       	sbiw	r26, 0x03	; 3
     e36:	14 96       	adiw	r26, 0x04	; 4
     e38:	ec 90       	ld	r14, X
     e3a:	14 97       	sbiw	r26, 0x04	; 4
     e3c:	15 96       	adiw	r26, 0x05	; 5
     e3e:	fc 90       	ld	r15, X
     e40:	15 97       	sbiw	r26, 0x05	; 5
     e42:	16 96       	adiw	r26, 0x06	; 6
     e44:	0c 91       	ld	r16, X
     e46:	16 97       	sbiw	r26, 0x06	; 6
     e48:	17 96       	adiw	r26, 0x07	; 7
     e4a:	1c 91       	ld	r17, X
     e4c:	17 97       	sbiw	r26, 0x07	; 7
     e4e:	0e 94 d5 0b 	call	0x17aa	; 0x17aa <__subdi3>
     e52:	c1 cf       	rjmp	.-126    	; 0xdd6 <tg_packed_dot+0xa2>
     e54:	e3 30       	cpi	r30, 0x03	; 3
     e56:	09 f0       	breq	.+2      	; 0xe5a <tg_packed_dot+0x126>
     e58:	be cf       	rjmp	.-132    	; 0xdd6 <tg_packed_dot+0xa2>
     e5a:	8f ef       	ldi	r24, 0xFF	; 255
     e5c:	9f ef       	ldi	r25, 0xFF	; 255
     e5e:	cc cf       	rjmp	.-104    	; 0xdf8 <tg_packed_dot+0xc4>


Kernel calls/branches: ['__adddi3', '__subdi3', 'tg_packed_dot+0x68', 'tg_packed_dot+0xa2', 'tg_packed_dot+0xc4']

Integer helper:
000017aa <__subdi3>:
    17aa:	2a 19       	sub	r18, r10
    17ac:	3b 09       	sbc	r19, r11
    17ae:	4c 09       	sbc	r20, r12
    17b0:	5d 09       	sbc	r21, r13
    17b2:	6e 09       	sbc	r22, r14
    17b4:	7f 09       	sbc	r23, r15
    17b6:	80 0b       	sbc	r24, r16
    17b8:	91 0b       	sbc	r25, r17
    17ba:	08 95       	ret


Integer helper:
00001798 <__adddi3>:
    1798:	2a 0d       	add	r18, r10
    179a:	3b 1d       	adc	r19, r11
    179c:	4c 1d       	adc	r20, r12
    179e:	5d 1d       	adc	r21, r13
    17a0:	6e 1d       	adc	r22, r14
    17a2:	7f 1d       	adc	r23, r15
    17a4:	80 1f       	adc	r24, r16
    17a6:	91 1f       	adc	r25, r17
    17a8:	08 95       	ret


PASS: AVR packed-dot and its integer add/sub helpers contain no multiply or floating-point helper calls.

Frozen model/scaler constants in flash (.text):
00000181 l     O .text	00000050 nano_mean
00000131 l     O .text	00000050 nano_scale
000005e1 l     O .text	00000004 tg_rmsnorm_eps_0
000005e5 l     O .text	00000050 tg_rmsnorm_gamma_0
00000635 l     O .text	00000004 tg_scale_0
00000639 l     O .text	00000100 tg_bias_0
00000739 l     O .text	00000140 tg_weights_0
00000259 l     O .text	00000004 tg_rmsnorm_eps_1
0000025d l     O .text	00000100 tg_rmsnorm_gamma_1
0000035d l     O .text	00000004 tg_scale_1
00000361 l     O .text	00000080 tg_bias_1
000003e1 l     O .text	00000200 tg_weights_1
000001d1 l     O .text	00000004 tg_scale_2
000001d5 l     O .text	0000002c tg_bias_2
00000201 l     O .text	00000058 tg_weights_2

Linked heap allocator symbols: []
```

## Full host precision validation

```text
$ python3 engine-arduino/prepare_validation.py --output /private/tmp/ternaryguard-phase5
ACTIVE ARTIFACT VERIFICATION {"mode": "verify-active", "hashes": {"checkpoint_sha256": "c335f7d468c932c3855b190061f452ff2b5b0f3fca8276bf1bff6d674ad13774", "header_sha256": "e1d919a0049e2c492d4ce54168cf9cb223bd90b4b1b74554539089ccf4c60ee8", "preprocessing_sha256": "1339886eaa7cde612b32831057afb225b7d5f314f8ebecdc6f8d57ae99c55c7d"}, "preprocessing_hash_format": "canonical JSON SHA-256", "header": "/Users/aryanshukla/Desktop/MAIN/02_Projects/Active/TernaryGuard/model_weights.h", "samples": 69040, "prediction_matches": 69040, "prediction_disagreements": 0, "max_abs_logit_error": 1.1444091796875e-05, "atol": 2e-05, "rtol": 2e-05, "artifacts_rewritten": false}
$ cc -std=c11 -O2 -Wall -Wextra -Werror -pedantic -ffp-contract=off -fno-fast-math -fstack-usage -I/private/tmp/ternaryguard-phase5/reference/build -I/Users/aryanshukla/Desktop/MAIN/02_Projects/Active/TernaryGuard /Users/aryanshukla/Desktop/MAIN/02_Projects/Active/TernaryGuard/engine-software/ternary_infer.c /Users/aryanshukla/Desktop/MAIN/02_Projects/Active/TernaryGuard/engine-software/runner.c -lm -o /private/tmp/ternaryguard-phase5/reference/build/ternary_infer
$ /private/tmp/ternaryguard-phase5/reference/build/ternary_infer --preprocess /private/tmp/ternaryguard-phase5/reference/test_raw_f64.bin /private/tmp/ternaryguard-phase5/reference/preprocessed_f32.bin
Processed 69040 rows (preprocessing)
$ /private/tmp/ternaryguard-phase5/reference/build/ternary_infer --run /private/tmp/ternaryguard-phase5/reference/test_raw_f64.bin /private/tmp/ternaryguard-phase5/reference/logits_f32.bin
Processed 69040 rows (inference)
{
  "samples": 69040,
  "prediction_matches": 69040,
  "prediction_disagreements": 0,
  "max_abs_preprocessing_error": 0.0,
  "max_abs_logit_error": 1.1444091796875e-05,
  "logits_within_export_tolerance": true,
  "c_accuracy": 0.8108198146002318,
  "torch_accuracy": 0.8108198146002318,
  "c_macro_f1": 0.78209432162603,
  "torch_macro_f1": 0.78209432162603,
  "bashlite_tcp": {
    "total": 5555,
    "c_true_positives": 1,
    "torch_true_positives": 1
  },
  "row_identity_sha256": "b3d5c16fceb383423cdab2982d5e119280bdbed8d72246235f798e56d51be1f1",
  "frozen_artifact_hashes": {
    "checkpoint_sha256": "c335f7d468c932c3855b190061f452ff2b5b0f3fca8276bf1bff6d674ad13774",
    "header_sha256": "e1d919a0049e2c492d4ce54168cf9cb223bd90b4b1b74554539089ccf4c60ee8",
    "preprocessing_sha256": "1339886eaa7cde612b32831057afb225b7d5f314f8ebecdc6f8d57ae99c55c7d"
  }
}
$ cc -std=c11 -O2 -Wall -Wextra -Werror -ffp-contract=off -fno-fast-math -shared -fPIC -I/Users/aryanshukla/Desktop/MAIN/02_Projects/Active/TernaryGuard/engine-arduino/include -I/private/tmp/ternaryguard-phase5/generated /Users/aryanshukla/Desktop/MAIN/02_Projects/Active/TernaryGuard/engine-arduino/src/ternary_avr.c -lm -o /private/tmp/ternaryguard-phase5/nano-host.so
{
  "scope": "HOST build of AVR source using host libm; NOT an AVR simulator or physical board",
  "preprocessing": "on-device float32 candidate; hardware acceptance pending",
  "samples": 69040,
  "prediction_disagreements": 0,
  "mismatch_indices": [],
  "max_preprocessing_error": 1.430511474609375e-06,
  "max_logit_error": 2.288818359375e-05,
  "tcp_total": 5555,
  "tcp_true_positives": 1,
  "proposed_stratified_rows": 330,
  "proposed_tcp_rows": 5555,
  "sole_tcp_hit_test_index": 59782
}

```

## Full test suite

```text
$ python3 -m pytest model/tests/ -v --tb=short
============================= test session starts ==============================
platform darwin -- Python 3.12.7, pytest-7.4.4, pluggy-1.0.0 -- /opt/anaconda3/bin/python3
cachedir: .pytest_cache
rootdir: /Users/aryanshukla/Desktop/MAIN/02_Projects/Active/TernaryGuard
plugins: anyio-4.2.0
collecting ... collected 99 items

model/tests/test_active_artifact_gates.py::test_verify_active_cli_catches_corrupted_header_and_never_rewrites PASSED [  1%]
model/tests/test_active_artifact_gates.py::test_verify_active_rejects_other_hash_mismatches[checkpoint] PASSED [  2%]
model/tests/test_active_artifact_gates.py::test_verify_active_rejects_other_hash_mismatches[preprocessing] PASSED [  3%]
model/tests/test_active_artifact_gates.py::test_verify_active_evaluates_existing_header_even_if_hash_is_updated PASSED [  4%]
model/tests/test_active_artifact_gates.py::test_legacy_plot_cli_refuses_invalidated_csv PASSED [  5%]
model/tests/test_active_artifact_gates.py::test_legacy_plot_cli_flag_renders_warning_into_pngs PASSED [  6%]
model/tests/test_c_engine.py::test_integer_dot_all_codes_unaligned_rows_and_large_signed_values PASSED [  7%]
model/tests/test_c_engine.py::test_integer_dot_rejects_reserved_code_and_oversized_input PASSED [  8%]
model/tests/test_c_engine.py::test_preprocessing_signed_values_and_nonfinite_rejection PASSED [  9%]
model/tests/test_c_engine.py::test_frozen_golden_vectors_match_logits_and_predictions PASSED [ 10%]
model/tests/test_c_engine.py::test_cli_rejects_partial_raw_record PASSED [ 11%]
model/tests/test_data_pipeline.py::TestFeatureNames::test_count PASSED   [ 12%]
model/tests/test_data_pipeline.py::TestFeatureNames::test_unique PASSED  [ 13%]
model/tests/test_data_pipeline.py::TestFeatureNames::test_format PASSED  [ 14%]
model/tests/test_data_pipeline.py::TestLabelNames::test_count PASSED     [ 15%]
model/tests/test_data_pipeline.py::TestLabelNames::test_benign_is_zero PASSED [ 16%]
model/tests/test_data_pipeline.py::TestLabelNames::test_all_keys_present PASSED [ 17%]
model/tests/test_data_pipeline.py::TestLabelDetection::test_benign PASSED [ 18%]
model/tests/test_data_pipeline.py::TestLabelDetection::test_mirai_ack PASSED [ 19%]
model/tests/test_data_pipeline.py::TestLabelDetection::test_mirai_udpplain PASSED [ 20%]
model/tests/test_data_pipeline.py::TestLabelDetection::test_mirai_udp PASSED [ 21%]
model/tests/test_data_pipeline.py::TestLabelDetection::test_bashlite_combo PASSED [ 22%]
model/tests/test_data_pipeline.py::TestLabelDetection::test_gafgyt_alias PASSED [ 23%]
model/tests/test_data_pipeline.py::TestLabelDetection::test_unknown PASSED [ 24%]
model/tests/test_data_pipeline.py::TestDatasetLoading::test_missing_directory_returns_empty PASSED [ 25%]
model/tests/test_data_pipeline.py::TestDatasetLoading::test_load_synthetic_csvs PASSED [ 26%]
model/tests/test_data_pipeline.py::TestPreprocessing::test_scaler_output_shape PASSED [ 27%]
model/tests/test_data_pipeline.py::TestPreprocessing::test_scaler_zero_mean PASSED [ 28%]
model/tests/test_data_pipeline.py::TestPreprocessing::test_scaler_unit_variance PASSED [ 29%]
model/tests/test_data_pipeline.py::TestPreprocessing::test_transform_without_fit PASSED [ 30%]
model/tests/test_data_pipeline.py::TestSplits::test_split_proportions PASSED [ 31%]
model/tests/test_data_pipeline.py::TestSplits::test_labels_preserved PASSED [ 32%]
model/tests/test_data_pipeline.py::TestSplits::test_split_types PASSED   [ 33%]
model/tests/test_data_pipeline.py::TestDataLoaders::test_loader_creation PASSED [ 34%]
model/tests/test_data_pipeline.py::TestDataLoaders::test_loader_batch_shape PASSED [ 35%]
model/tests/test_data_pipeline.py::TestFeatureConfig::test_save_load_roundtrip PASSED [ 36%]
model/tests/test_data_pipeline.py::TestFeatureSelection::test_mutual_info_count PASSED [ 37%]
model/tests/test_data_pipeline.py::TestFeatureSelection::test_variance_count PASSED [ 38%]
model/tests/test_data_pipeline.py::TestFeatureSelection::test_correlation_count PASSED [ 39%]
model/tests/test_data_pipeline.py::TestFeatureSelection::test_invalid_method_raises PASSED [ 40%]
model/tests/test_nano_port.py::test_nano_output_alias_matches_frozen_golden_vectors PASSED [ 41%]
model/tests/test_nano_port.py::test_nano_float32_preprocessing_signed_tiny_values_and_nonfinite PASSED [ 42%]
model/tests/test_nano_port.py::test_serial_protocol_crc_sequence_status_and_lengths PASSED [ 43%]
model/tests/test_phase3_integrity.py::test_schema_known_positions PASSED [ 44%]
model/tests/test_phase3_integrity.py::test_sampling_covers_late_devices_and_rejects_bad_headers PASSED [ 45%]
model/tests/test_phase3_integrity.py::test_preprocessing_never_fits_heldout_rows[identity] PASSED [ 46%]
model/tests/test_phase3_integrity.py::test_preprocessing_never_fits_heldout_rows[signed_log1p] PASSED [ 47%]
model/tests/test_phase3_integrity.py::test_export_preserves_noninteger_bias_logits_and_exact_float_constants[False] PASSED [ 48%]
model/tests/test_phase3_integrity.py::test_export_preserves_noninteger_bias_logits_and_exact_float_constants[True] PASSED [ 49%]
model/tests/test_phase3_integrity.py::test_export_only_loads_frozen_bundle_without_dataset PASSED [ 50%]
model/tests/test_phase3_integrity.py::test_results_quotes_and_schema PASSED [ 51%]
model/tests/test_phase3_integrity.py::test_per_layer_padding_and_fp_classifier_budget PASSED [ 52%]
model/tests/test_phase3_integrity.py::test_heldout_devices_require_disjoint_sets_and_class_coverage PASSED [ 53%]
model/tests/test_phase3_integrity.py::test_feature_selection_does_not_spend_slots_on_duplicates PASSED [ 54%]
model/tests/test_phase3_integrity.py::test_golden_vectors_include_every_class PASSED [ 55%]
model/tests/test_phase3_integrity.py::test_header_compiles_as_c_and_constants_have_expected_sizes PASSED [ 56%]
model/tests/test_phase3_integrity.py::test_signed_log_transform_handles_negative_covariances_and_large_values PASSED [ 57%]
model/tests/test_phase3_integrity.py::test_scaler_config_rejects_feature_dimension_mismatch PASSED [ 58%]
model/tests/test_phase3_integrity.py::test_preprocessing_hash_survives_json_integer_label_keys PASSED [ 59%]
model/tests/test_phase3_integrity.py::test_frozen_rows_and_source_content_are_verified PASSED [ 60%]
model/tests/test_ternary_linear.py::TestTernaryQuantize::test_output_values_are_ternary PASSED [ 61%]
model/tests/test_ternary_linear.py::TestTernaryQuantize::test_scale_is_absmean PASSED [ 62%]
model/tests/test_ternary_linear.py::TestTernaryQuantize::test_threshold_logic PASSED [ 63%]
model/tests/test_ternary_linear.py::TestTernaryQuantize::test_ste_gradient_identity PASSED [ 64%]
model/tests/test_ternary_linear.py::TestTernaryQuantize::test_gradient_is_nonzero PASSED [ 65%]
model/tests/test_ternary_linear.py::TestTernaryQuantize::test_all_zeros_weight PASSED [ 66%]
model/tests/test_ternary_linear.py::TestTernaryQuantize::test_all_same_sign PASSED [ 67%]
model/tests/test_ternary_linear.py::TestTernaryLinear::test_output_shape_2d PASSED [ 68%]
model/tests/test_ternary_linear.py::TestTernaryLinear::test_output_shape_single_sample PASSED [ 69%]
model/tests/test_ternary_linear.py::TestTernaryLinear::test_output_shape_no_bias PASSED [ 70%]
model/tests/test_ternary_linear.py::TestTernaryLinear::test_bias_parameter_registration PASSED [ 71%]
model/tests/test_ternary_linear.py::TestTernaryLinear::test_gradients_nonzero_weight PASSED [ 72%]
model/tests/test_ternary_linear.py::TestTernaryLinear::test_gradients_nonzero_bias PASSED [ 73%]
model/tests/test_ternary_linear.py::TestTernaryLinear::test_weight_distribution_balanced PASSED [ 74%]
model/tests/test_ternary_linear.py::TestTernaryLinear::test_quantized_weights_are_int8_ternary PASSED [ 75%]
model/tests/test_ternary_linear.py::TestTernaryLinear::test_scale_is_positive PASSED [ 76%]
model/tests/test_ternary_linear.py::TestTernaryLinear::test_deterministic_forward PASSED [ 77%]
model/tests/test_ternary_linear.py::TestTernaryLinear::test_repr PASSED  [ 78%]
model/tests/test_ternary_linear.py::TestRMSNorm::test_output_shape PASSED [ 79%]
model/tests/test_ternary_linear.py::TestRMSNorm::test_rms_near_one PASSED [ 80%]
model/tests/test_ternary_linear.py::TestRMSNorm::test_gradient_flows PASSED [ 81%]
model/tests/test_ternary_linear.py::TestRMSNorm::test_gamma_gradient PASSED [ 82%]
model/tests/test_ternary_linear.py::TestRMSNorm::test_identity_at_unit_rms PASSED [ 83%]
model/tests/test_ternary_linear.py::TestTernaryMLP::test_forward_shape PASSED [ 84%]
model/tests/test_ternary_linear.py::TestTernaryMLP::test_forward_multiclass PASSED [ 85%]
model/tests/test_ternary_linear.py::TestTernaryMLP::test_training_reduces_loss PASSED [ 86%]
model/tests/test_ternary_linear.py::TestTernaryMLP::test_count_parameters PASSED [ 87%]
model/tests/test_ternary_linear.py::TestTernaryMLP::test_packed_size_below_flash PASSED [ 88%]
model/tests/test_ternary_linear.py::TestTernaryMLP::test_no_rmsnorm PASSED [ 89%]
model/tests/test_ternary_linear.py::TestTernaryMLP::test_fp_output_layer PASSED [ 90%]
model/tests/test_ternary_linear.py::TestTernaryMLP::test_single_hidden_layer PASSED [ 91%]
model/tests/test_ternary_linear.py::TestTernaryMLP::test_deep_model PASSED [ 92%]
model/tests/test_ternary_linear.py::TestTernaryMLP::test_estimate_packed_size_includes_rmsnorm PASSED [ 93%]
model/tests/test_ternary_linear.py::TestTernaryMLP::test_quantize_paths_agree PASSED [ 94%]
model/tests/test_ternary_linear.py::TestTernaryMLP::test_ram_fits_2kb PASSED [ 95%]
model/tests/test_ternary_linear.py::TestTernaryMLP::test_pack_ternary_2bit_roundtrip PASSED [ 96%]
model/tests/test_ternary_linear.py::TestTernaryMLP::test_export_weights_header PASSED [ 97%]
model/tests/test_ternary_linear.py::TestTernaryMLP::test_pack_row_major_not_transposed PASSED [ 98%]
model/tests/test_ternary_linear.py::TestTernaryMLP::test_decode_matches_spec_snippet PASSED [100%]

============================= 99 passed in 42.63s ==============================

```
