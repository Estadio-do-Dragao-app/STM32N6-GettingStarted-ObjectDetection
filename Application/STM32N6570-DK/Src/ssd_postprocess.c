#include "ssd_postprocess.h"

#include <math.h>
#include <stdbool.h>
#include <string.h>

#define SSDLITE_SCORE_SCALE  (0.024268511682748795f)
#define SSDLITE_SCORE_ZERO   (-1.0f)
#define SSDLITE_BOX_SCALE    (0.09016282856464386f)
#define SSDLITE_BOX_ZERO     (74.0f)

#define SSDLITE_WX 10.0f
#define SSDLITE_WY 10.0f
#define SSDLITE_WW 5.0f
#define SSDLITE_WH 5.0f

typedef struct {
  float cx;
  float cy;
  float w;
  float h;
} prior_t;

static prior_t g_priors[APP_SSDLITE_PRIOR_COUNT];
static bool g_priors_ready = false;

static float g_tmp_scores[APP_SSDLITE_PRIOR_COUNT];
static ssd_det_t g_tmp_dets[APP_SSDLITE_PRIOR_COUNT];

static inline float clamp01(float v)
{
  if (v < 0.0f) return 0.0f;
  if (v > 1.0f) return 1.0f;
  return v;
}

static inline float iou_xyxy(const ssd_det_t *a, const ssd_det_t *b)
{
  const float xx1 = (a->x1 > b->x1) ? a->x1 : b->x1;
  const float yy1 = (a->y1 > b->y1) ? a->y1 : b->y1;
  const float xx2 = (a->x2 < b->x2) ? a->x2 : b->x2;
  const float yy2 = (a->y2 < b->y2) ? a->y2 : b->y2;

  const float w = (xx2 > xx1) ? (xx2 - xx1) : 0.0f;
  const float h = (yy2 > yy1) ? (yy2 - yy1) : 0.0f;
  const float inter = w * h;

  const float area_a = (a->x2 - a->x1) * (a->y2 - a->y1);
  const float area_b = (b->x2 - b->x1) * (b->y2 - b->y1);
  const float uni = area_a + area_b - inter;

  if (uni <= 1e-6f) {
    return 0.0f;
  }
  return inter / uni;
}

static void build_priors(void)
{
  static const int fmap[] = {19, 10, 5, 3, 2, 1};
  static const float scales[] = {0.20f, 0.34f, 0.48f, 0.62f, 0.76f, 0.90f, 1.04f};
  static const float ars[] = {2.0f, 3.0f};

  uint32_t idx = 0;
  for (uint32_t k = 0; k < 6; k++) {
    const int f = fmap[k];
    const float s = scales[k];
    const float s_next = scales[k + 1];

    for (int y = 0; y < f; y++) {
      for (int x = 0; x < f; x++) {
        const float cx = ((float)x + 0.5f) / (float)f;
        const float cy = ((float)y + 0.5f) / (float)f;

        g_priors[idx++] = (prior_t){cx, cy, s, s};

        const float s_prime = sqrtf(s * s_next);
        g_priors[idx++] = (prior_t){cx, cy, s_prime, s_prime};

        for (uint32_t a = 0; a < 2; a++) {
          const float ar_sqrt = sqrtf(ars[a]);
          g_priors[idx++] = (prior_t){cx, cy, s * ar_sqrt, s / ar_sqrt};
          g_priors[idx++] = (prior_t){cx, cy, s / ar_sqrt, s * ar_sqrt};
        }
      }
    }
  }

  g_priors_ready = (idx == APP_SSDLITE_PRIOR_COUNT);
}

void ssd_post_init(void)
{
  if (!g_priors_ready) {
    build_priors();
  }
}

static uint32_t pick_candidates(const int8_t *scores_q, uint32_t *cand_idx, uint32_t cand_max)
{
  uint32_t n = 0;

  for (uint32_t i = 0; i < APP_SSDLITE_PRIOR_COUNT; i++) {
    const int8_t q_bg = scores_q[(i * 2U) + 0U];
    const int8_t q_p = scores_q[(i * 2U) + 1U];

    const float logit_bg = (((float)q_bg) - SSDLITE_SCORE_ZERO) * SSDLITE_SCORE_SCALE;
    const float logit_p = (((float)q_p) - SSDLITE_SCORE_ZERO) * SSDLITE_SCORE_SCALE;

    const float m = (logit_bg > logit_p) ? logit_bg : logit_p;
    const float eb = expf(logit_bg - m);
    const float ep = expf(logit_p - m);
    const float p_person = ep / (eb + ep + 1e-8f);

    if (p_person >= APP_SSDLITE_CONF_THRESH) {
      if (n < cand_max) {
        cand_idx[n] = i;
        g_tmp_scores[n] = p_person;
      }
      n++;
    }
  }

  return (n > cand_max) ? cand_max : n;
}

static uint32_t decode_candidates(
  const int8_t *boxes_q,
  const uint32_t *cand_idx,
  uint32_t cand_count,
  ssd_det_t *decoded,
  uint32_t frame_w,
  uint32_t frame_h)
{
  uint32_t n = 0;

  for (uint32_t j = 0; j < cand_count; j++) {
    const uint32_t i = cand_idx[j];

    const float dx = ((((float)boxes_q[(i * 4U) + 0U]) - SSDLITE_BOX_ZERO) * SSDLITE_BOX_SCALE) / SSDLITE_WX;
    const float dy = ((((float)boxes_q[(i * 4U) + 1U]) - SSDLITE_BOX_ZERO) * SSDLITE_BOX_SCALE) / SSDLITE_WY;
    const float dw = ((((float)boxes_q[(i * 4U) + 2U]) - SSDLITE_BOX_ZERO) * SSDLITE_BOX_SCALE) / SSDLITE_WW;
    const float dh = ((((float)boxes_q[(i * 4U) + 3U]) - SSDLITE_BOX_ZERO) * SSDLITE_BOX_SCALE) / SSDLITE_WH;

    const prior_t p = g_priors[i];

    const float cx = dx * p.w + p.cx;
    const float cy = dy * p.h + p.cy;
    const float bw = expf(dw) * p.w;
    const float bh = expf(dh) * p.h;

    const float x1 = clamp01(cx - (bw * 0.5f));
    const float y1 = clamp01(cy - (bh * 0.5f));
    const float x2 = clamp01(cx + (bw * 0.5f));
    const float y2 = clamp01(cy + (bh * 0.5f));

    ssd_det_t d;
    d.x1 = x1 * (float)frame_w;
    d.y1 = y1 * (float)frame_h;
    d.x2 = x2 * (float)frame_w;
    d.y2 = y2 * (float)frame_h;
    d.score = g_tmp_scores[j];

    if (d.x2 > d.x1 && d.y2 > d.y1) {
      decoded[n++] = d;
    }
  }

  return n;
}

static uint32_t nms(const ssd_det_t *in, uint32_t n, ssd_det_t *out, uint32_t out_max)
{
  bool suppressed[APP_SSDLITE_PRIOR_COUNT] = {0};
  uint32_t out_n = 0;

  for (uint32_t step = 0; step < n; step++) {
    float best_score = -1.0f;
    int32_t best_idx = -1;

    for (uint32_t i = 0; i < n; i++) {
      if (!suppressed[i] && in[i].score > best_score) {
        best_score = in[i].score;
        best_idx = (int32_t)i;
      }
    }

    if (best_idx < 0) {
      break;
    }

    const ssd_det_t best = in[(uint32_t)best_idx];
    suppressed[(uint32_t)best_idx] = true;

    if (out_n < out_max) {
      out[out_n++] = best;
    }

    for (uint32_t i = 0; i < n; i++) {
      if (!suppressed[i]) {
        const float ov = iou_xyxy(&best, &in[i]);
        if (ov >= APP_SSDLITE_NMS_IOU) {
          suppressed[i] = true;
        }
      }
    }
  }

  return out_n;
}

uint32_t ssd_post_process(
  const int8_t *scores_q,
  const int8_t *boxes_q,
  uint32_t frame_w,
  uint32_t frame_h,
  ssd_det_t *out_dets,
  uint32_t out_max)
{
  uint32_t cand_idx[APP_SSDLITE_PRIOR_COUNT];

  if (!g_priors_ready) {
    ssd_post_init();
  }

  const uint32_t cand_count = pick_candidates(scores_q, cand_idx, APP_SSDLITE_PRIOR_COUNT);
  if (cand_count == 0) {
    return 0;
  }

  const uint32_t decoded_n = decode_candidates(boxes_q, cand_idx, cand_count, g_tmp_dets, frame_w, frame_h);
  if (decoded_n == 0) {
    return 0;
  }

  return nms(g_tmp_dets, decoded_n, out_dets, out_max);
}

uint32_t ssd_build_grid_from_dets(
  const ssd_det_t *dets,
  uint32_t det_count,
  grid_point_t *grid,
  uint32_t grid_max)
{
  uint32_t n = 0;
  for (uint32_t i = 0; i < det_count && n < grid_max; i++) {
    const int x = (int)((dets[i].x1 + dets[i].x2) * 0.5f);
    const int y = (int)(dets[i].y2);
    grid[n].x = x;
    grid[n].y = y;
    grid[n].count = 1;
    n++;
  }
  return n;
}
