/* Phase 5 AVR port of the frozen Phase 4 integer engine. */
#include "nano_infer.h"
#include "../../model_weights.h"
#include "nano_preprocessing.h"
#include <math.h>
#include <string.h>
#ifdef __AVR__
#define READ_BYTE(p) pgm_read_byte(p)
#define READ_FLOAT(p) pgm_read_float(p)
_Static_assert(sizeof(double) == 4, "This build validates AVR 32-bit double");
#else
#define READ_BYTE(p) (*(p))
#define READ_FLOAT(p) (*(p))
#endif
_Static_assert(sizeof(float) == 4, "binary32 required");
/* avr-libc 2.0 does not provide log1p. Correct log(1+x) for the rounding
 * in 1+x; return x directly when that addition rounds to one. */
static float nano_log1p(float x) {
    float u = 1.0f + x;
    return u == 1.0f ? x : logf(u) * (x / (u - 1.0f));
}
int nano_preprocess(float values[TG_FEATURES]) {
    for (uint8_t i = 0; i < TG_FEATURES; ++i) {
        float x = values[i];
        if (!isfinite(x)) return -1;
        float logged = copysignf(nano_log1p(fabsf(x)), x);
        values[i] = (logged - READ_FLOAT(nano_mean+i)) / READ_FLOAT(nano_scale+i);
        if (!isfinite(values[i])) return -1;
    }
    return 0;
}
/* Keep the real firmware kernel inspectable under link-time optimization. */
#if defined(__GNUC__)
__attribute__((noinline))
#endif
int tg_packed_dot(const uint8_t *packed, uint16_t offset, const int64_t *x,
                  uint16_t count, int64_t *result) {
    int64_t sum = 0;
    if (count > TG_MAX_WIDTH) return -1;
    for (uint16_t i = 0; i < count; ++i, ++offset) {
        uint8_t code = (uint8_t)((READ_BYTE(packed + (offset >> 2)) >>
                                 ((offset & 3u) << 1)) & 3u);
        if (code == 1u) sum += x[i];
        else if (code == 2u) sum -= x[i];
        else if (code == 3u) return -1;
    }
    *result = sum;
    return 0;
}

/* Convert once per layer to a shared binary exponent. Every |fixed[i]| < 2^40;
 * 64 terms give |sum| < 2^46, far below INT64_MAX. Conversion truncation error
 * per input is < 2^(exponent-40). This is numerical arithmetic, not retraining
 * or a new stored weight quantization. No floating point occurs in packed_dot. */
static int to_fixed(const float *input, uint16_t count, int64_t *fixed, int *shift) {
    float largest = 0.0f;
    for (uint16_t i = 0; i < count; ++i) {
        if (!isfinite(input[i])) return -1;
        if (fabsf(input[i]) > largest) largest = fabsf(input[i]);
    }
    int exponent = 0;
    (void)frexpf(largest, &exponent);
    *shift = 40 - exponent;
    for (uint16_t i = 0; i < count; ++i)
        fixed[i] = (int64_t)ldexpf(input[i], *shift);
    return 0;
}

static int layer(tg_workspace *work, float *input, float *output,
                 uint16_t ins, uint16_t outs, const uint8_t *weights,
                 const float *bias, const float *scale, const float *gamma,
                 const float *epsilon, int relu) {
    if (gamma != NULL) {
        float squares = 0.0f;
        for (uint16_t i = 0; i < ins; ++i) squares += input[i] * input[i];
        float rms = sqrtf(squares / (float)ins + READ_FLOAT(epsilon));
        if (!isfinite(rms) || rms <= 0.0f) return -1;
        for (uint16_t i = 0; i < ins; ++i)
            input[i] = (input[i] / rms) * READ_FLOAT(gamma + i);
    }
    int shift;
    if (to_fixed(input, ins, work->fixed, &shift)) return -1;
    for (uint16_t row = 0, offset = 0; row < outs; ++row, offset += ins) {
        int64_t sum;
        if (tg_packed_dot(weights, offset, work->fixed, ins, &sum)) return -1;
        /* Restore the binary exponent, then the trained scale and float bias. */
        float value = ldexpf((float)sum, -shift) * READ_FLOAT(scale) + READ_FLOAT(bias + row);
        if (!isfinite(value)) return -1;
        output[row] = relu && value < 0.0f ? 0.0f : value;
    }
    return 0;
}

int tg_infer_preprocessed(tg_workspace *work, const float input[TG_FEATURES], float logits[TG_CLASSES]) {
    memmove(work->a, input, TG_FEATURES * sizeof(float));
    if (layer(work, work->a, work->b, 20, 64, tg_weights_0, tg_bias_0, &tg_scale_0,
              tg_rmsnorm_gamma_0, &tg_rmsnorm_eps_0, 1)) return -1;
    if (layer(work, work->b, work->a, 64, 32, tg_weights_1, tg_bias_1, &tg_scale_1,
              tg_rmsnorm_gamma_1, &tg_rmsnorm_eps_1, 1)) return -1;
    return layer(work, work->a, logits, 32, 11, tg_weights_2, tg_bias_2, &tg_scale_2,
                 NULL, NULL, 0);
}

