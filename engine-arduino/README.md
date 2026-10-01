# Phase 5 Nano port — physical validation passed

Frozen reference remains commit fad0f0c / model/active_model.json. This port is
validated on a physical Nano for 330 stratified rows and all 5,555 TCP rows
(5,855 unique rows). See [the hardware report](../docs/phase5_hardware_validation.md)
for complete raw evidence, precision scope, timing and SRAM measurements.

## Precision decision

The deployment runs signed-log1p and StandardScaler **on the Nano in
float32**, with float32 raw selected features sent over UART. Host column
selection uses the frozen 20 feature indices; the host does not log-transform
or normalize mode-1 packets. Scaler constants are generated from the frozen JSON
with explicit float32 rounding and stored in PROGMEM. The model header is included
directly from the repository root; its hashes are checked before building.

The installed AVR-GCC 7.3.0 / avr-libc 2.0 build asserts sizeof(double)==4.
That libc has no log1p declaration. nano_log1p computes logf(1+x) corrected for
rounding in 1+x, and returns x when 1+x rounds to one. The host build of this
exact source matched all 69,040 frozen predictions and preserved TCP 1/5,555.
This is a host-libm float32 test, **not proof of AVR-libc or physical-board parity**.
Physical execution subsequently passed on 5,855 unique rows, including the full
TCP class; the remaining rows have not been executed on AVR. A mode-2
preprocessed-input diagnostic exists to isolate preprocessing failures; it must
not be presented as a successful on-device preprocessing validation.

## Build and prepare

Requires PlatformIO, and Python with the existing model dependencies plus
pyserial for hardware checks. Paths below use a normal PlatformIO installation;
the session used an isolated toolchain under /private/tmp (see validation report).

```sh
python3 engine-arduino/prepare_validation.py --output /tmp/tg-phase5
pio run -d engine-arduino -e nano_new
```

Choose nano_old for a board with the old bootloader. USB interface chip alone
does not establish bootloader type. See the official
[Arduino processor selection guidance](https://support.arduino.cc/hc/en-us/articles/4401874304274-Select-the-right-processor-for-Arduino-Nano)
and [PlatformIO Nano definition](https://docs.platformio.org/en/latest/boards/atmelavr/nanoatmega328new.html).
Do not replace the frozen model to remedy any parity failure.

## Real hardware acceptance (completed for this scope)

Executed sample plan: 330 stratified rows (30/class,
seed 42), explicitly including TCP's sole true positive at frozen test index
59782. Separately test **all 5,555 TCP rows** to establish exact 1/5,555 recall.
This is 5,885 transactions across two passes, 5,855 unique rows, not full
69,040-row hardware coverage. The stratified subset alone cannot prove the
full TCP recall. Preparation regenerates source-verified frozen rows and compares
with the frozen checkpoint before selecting cases.

To repeat the test, replace PORT with its actual USB serial path (the validated
board used /dev/cu.usbserial-A5069RR4 and nano_new):

```sh
pio run -d engine-arduino -e nano_new -t upload --upload-port PORT
python3 engine-arduino/serial_validate.py --port PORT --cases /tmp/tg-phase5/serial_cases.npz --group subset --output /tmp/tg-phase5/hardware-subset.jsonl
python3 engine-arduino/serial_validate.py --port PORT --cases /tmp/tg-phase5/serial_cases.npz --group tcp --output /tmp/tg-phase5/hardware-tcp.jsonl
```

The protocol uses sequence numbers and CRC16 in both directions. Raw board reply
bytes and decoded records are logged. Invalid replies, firmware error status,
classification differences, or a TCP result other than 1/5,555 fail acceptance.
Never connect to Bluetooth/debug-console just because they are the only ports.

## Timing and RAM

D8/PB0 is high during preprocessing and inference, suitable for a logic analyzer.
The reported micros() interval excludes UART reception/transmission, includes
preprocessing, and runs with normal Arduino timer/serial interrupts enabled.
It is measured only when a real board replies. No host latency substitutes for it.

A startup canary paints unused SRAM above static data, leaving 32 bytes below
the startup stack unpainted. Replies report the smallest untouched bottom gap
observed since painting. The reported 2048-gap is a canary-based high-water
measurement, including static allocations and stack, not an exact cycle-by-cycle
peak. The initial reserved gap and marker coincidence limit precision. A gap
under 32 bytes fails the run. This check detects observed intrusion into the
painted gap; it does not by itself prove all possible interrupt/input paths safe.
The linked firmware has no heap allocator symbols. The 1,024-byte workspace is
static; UART input and output reuse its activation buffers.

Current avr-size static data is 1,212 bytes; the remaining 836 bytes are **not a
measured runtime safety margin**. Physical runs measured a 1,321-byte SRAM canary high-water mark with a 727-byte
untouched gap. Median micros() latency was 36.308 ms on the stratified subset
and 34.816 ms over TCP. Avrdude verified all 8,076 flashed bytes. These are
measured results subject to the canary limitations above. Phase 6 has not started.
