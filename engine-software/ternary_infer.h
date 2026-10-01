#ifndef TG_INFER_H
#define TG_INFER_H
#include <stddef.h>
#include <stdint.h>
#define TG_RAW_FEATURES 115
#define TG_FEATURES 20
#define TG_CLASSES 11
#define TG_MAX_WIDTH 64
/* Caller-owned, reusable, reentrant scratch; no heap allocation. */
typedef struct {
    float a[TG_MAX_WIDTH], b[TG_MAX_WIDTH];
    int64_t fixed[TG_MAX_WIDTH];
} tg_workspace;
int tg_preprocess(const double raw[TG_RAW_FEATURES], float selected[TG_FEATURES]);
int tg_infer_preprocessed(tg_workspace *work, const float input[TG_FEATURES], float logits[TG_CLASSES]);
int tg_infer(tg_workspace *work, const double raw[TG_RAW_FEATURES], float logits[TG_CLASSES]);
/* Integer-only packed dot product, also exposed for independent kernel tests.
 * Inputs must satisfy |x[i]| < 2^40 and count <= 64. Code 3 is rejected. */
int tg_packed_dot(const uint8_t *packed, uint16_t offset, const int64_t *x,
                  uint16_t count, int64_t *result);
size_t tg_model_bytes(void);
size_t tg_preprocessing_bytes(void);
#endif
