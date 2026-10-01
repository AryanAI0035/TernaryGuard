/* Phase 4 reference engine. Frozen weights are included directly, never copied. */
#include "ternary_infer.h"
#include "../model_weights.h"
#include "preprocessing.h"
#include <float.h>
#include <math.h>
#include <string.h>

#ifdef __AVR__
#define READ_BYTE(p) pgm_read_byte(p)
#define READ_FLOAT(p) pgm_read_float(p)
#define READ_DOUBLE(p) pgm_read_double(p)
#else
#define READ_BYTE(p) (*(p))
#define READ_FLOAT(p) (*(p))
#define READ_DOUBLE(p) (*(p))
#endif
_Static_assert(TG_INPUT_DIM == TG_FEATURES && TG_OUTPUT_DIM == TG_CLASSES &&
               TG_NUM_LAYERS == 3 && TG_USE_RMSNORM == 1 && TG_FORMAT_VERSION == 2,
               "Engine requires the frozen Phase 3 architecture");
_Static_assert(FLT_RADIX == 2 && FLT_MANT_DIG == 24, "Binary32 activations required");

int tg_preprocess(const double raw[TG_RAW_FEATURES], float selected[TG_FEATURES]) {
    for (uint16_t i = 0; i < TG_FEATURES; ++i) {
        double value = raw[READ_BYTE(tg_feature_indices + i)];
        if (!isfinite(value)) return -1;
        /* Same float64 log/scaler order as Python, then one float32 cast. */
        double logged = copysign(log1p(fabs(value)), value);
        selected[i] = (float)((logged - READ_DOUBLE(tg_feature_mean + i)) /
                              READ_DOUBLE(tg_feature_scale + i));
        if (!isfinite(selected[i])) return -1;
    }
    return 0;
}

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
        fixed[i] = (int64_t)ldexp((double)input[i], *shift);
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

int tg_infer(tg_workspace *work, const double raw[TG_RAW_FEATURES], float logits[TG_CLASSES]) {
    if (tg_preprocess(raw, work->a)) return -1;
    return tg_infer_preprocessed(work, work->a, logits);
}

size_t tg_model_bytes(void) {
    return sizeof(tg_weights_0) + sizeof(tg_weights_1) + sizeof(tg_weights_2) +
        sizeof(tg_bias_0) + sizeof(tg_bias_1) + sizeof(tg_bias_2) +
        sizeof(tg_scale_0) + sizeof(tg_scale_1) + sizeof(tg_scale_2) +
        sizeof(tg_rmsnorm_gamma_0) + sizeof(tg_rmsnorm_gamma_1) +
        sizeof(tg_rmsnorm_eps_0) + sizeof(tg_rmsnorm_eps_1) + sizeof(TG_DIMS);
}
size_t tg_preprocessing_bytes(void) {
    return sizeof(tg_feature_indices) + sizeof(tg_feature_mean) + sizeof(tg_feature_scale);
}
