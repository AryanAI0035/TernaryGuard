#define _POSIX_C_SOURCE 200809L
#include "ternary_infer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double seconds(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t)) exit(2);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}
static int benchmark(const char *path, const char *iterations_text) {
    /* Workstation-only input cache; excluded from engine scratch accounting. */
    static double rows[256][TG_RAW_FEATURES];
    tg_workspace work;
    float logits[TG_CLASSES];
    FILE *f = fopen(path, "rb");
    if (!f) return 2;
    size_t count = fread(rows, sizeof(rows[0]), 256, f);
    fclose(f);
    char *end;
    unsigned long iterations = strtoul(iterations_text, &end, 10);
    if (!count || *end || iterations < 256 || iterations > 100000000UL) return 2;
    volatile double checksum = 0.0;
    for (unsigned long i = 0; i < 4096; ++i)
        if (tg_infer(&work, rows[i % count], logits)) return 2;
    printf("{\"iterations_per_trial\":%lu,\"cached_rows\":%zu,\"microseconds_per_inference\":[", iterations, count);
    for (int trial = 0; trial < 7; ++trial) {
        double start = seconds();
        for (unsigned long i = 0; i < iterations; ++i) {
            if (tg_infer(&work, rows[i % count], logits)) return 2;
            checksum += logits[i % TG_CLASSES];
        }
        printf("%s%.6f", trial ? "," : "", (seconds() - start) * 1e6 / (double)iterations);
    }
    printf("],\"workspace_bytes\":%zu,\"raw_input_bytes\":%zu,\"output_bytes\":%zu,"
           "\"model_constant_bytes\":%zu,\"preprocessing_constant_bytes\":%zu,\"checksum\":%.9g}\n",
           sizeof(work), sizeof(rows[0]), sizeof(logits), tg_model_bytes(), tg_preprocessing_bytes(), checksum);
    return 0;
}
int main(int argc, char **argv) {
    if (argc == 4 && strcmp(argv[1], "--benchmark") == 0)
        return benchmark(argv[2], argv[3]);
    if (argc != 4 || (strcmp(argv[1], "--run") && strcmp(argv[1], "--preprocess"))) {
        fprintf(stderr, "Usage: %s --run|--preprocess RAW_F64 OUTPUT_F32\n       %s --benchmark RAW_F64 ITERATIONS\n", argv[0], argv[0]);
        return 2;
    }
    FILE *input = fopen(argv[2], "rb"), *output = fopen(argv[3], "wb");
    if (!input || !output) return 2;
    tg_workspace work;
    double raw[TG_RAW_FEATURES];
    float values[TG_FEATURES];
    int preprocessing = strcmp(argv[1], "--preprocess") == 0;
    size_t size = preprocessing ? TG_FEATURES : TG_CLASSES;
    size_t count, rows = 0;
    while ((count = fread(raw, 1, sizeof(raw), input)) != 0) {
        if (count != sizeof(raw) || (preprocessing ? tg_preprocess(raw, values) : tg_infer(&work, raw, values))) {
            fprintf(stderr, "Invalid input at row %zu\n", rows);
            return 1;
        }
        if (fwrite(values, sizeof(float), size, output) != size) return 2;
        ++rows;
    }
    if (ferror(input) || fclose(output)) return 2;
    fclose(input);
    fprintf(stderr, "Processed %zu rows (%s)\n", rows, preprocessing ? "preprocessing" : "inference");
    return 0;
}
