# Résumé and portfolio wording

## Project entry

**TernaryGuard — Embedded ML for IoT Botnet Classification**

Python, PyTorch, C, AVR-GCC, Arduino Nano, PlatformIO

[GitHub repository](https://github.com/AryanAI0035/TernaryGuard)

- Built a 20→64→32→11 ternary neural network with 2-bit packed weights and a 1,696-byte parameter footprint, 9.13× smaller than its FP32 counterpart; achieved 81.08% accuracy and 0.7821 macro-F1 on 69,040 held-out-device N-BaIoT samples.
- Implemented integer add/subtract/skip inference in C and AVR, with zero prediction disagreements against PyTorch across the full workstation test set and 5,855 unique test rows executed on a physical Arduino Nano.
- Deployed Nano firmware using PROGMEM in 8,076 bytes of flash and 1,212 bytes of static SRAM; measured 36.308 ms median inference on a 330-row stratified run, including on-device preprocessing.

If space allows only one bullet:

> Built and physically validated a compact ternary ML classifier on Arduino Nano: 1,696-byte model parameters, 8,076-byte firmware, 36.308 ms measured inference, and exact prediction parity on 5,855 unique hardware test rows.

## Description for a portfolio or GitHub profile

TernaryGuard runs one compact IoT botnet classifier in PyTorch, C and Arduino Nano firmware. The project includes evaluation on a held-out device, checked model exports, full-test-set C comparisons and recorded Nano measurements. FPGA acceleration is future work.

## Details to explain in interviews

- The 1,696-byte figure is model parameters, not firmware flash; 1,712 bytes includes exported header metadata. Static SRAM is not runtime peak: the observed canary high-water mark was approximately 1,321 bytes.
- The ternary dot product uses integer add/subtract/skip. Scaling, RMSNorm and preprocessing still use floating point. Do not describe the whole network as floating-point-free or multiplication-free.
- Workstation coverage is all 69,040 rows. Physical Nano coverage is 330 stratified rows plus all 5,555 TCP rows, overlapping in 30 cases: 5,855 unique rows. Hardware inference excludes UART time.
- BASHLITE TCP recall is 1/5,555 (0.018%), with the sole detection at index 59782. This prevents a complete 11-class production-detection claim. Be explicit about the tradeoff between compression and detection quality.
- There is no implemented live packet feature extractor or blocking system. Inputs are precomputed traffic statistics.
- FPGA has local simulation work, but actual Vivado acceptance remains pending. Do not list it as a completed FPGA deployment, timing-closed accelerator or measured speedup.

Sources: [model corrections](phase3_corrections.md), [final model metrics](benchmarks/phase3_final_seed42/ternary_2bit.json), [C measurements](phase4_validation.md), [physical Nano evidence](phase5_hardware_validation.md).
