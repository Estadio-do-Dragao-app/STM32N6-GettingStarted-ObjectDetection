#ifndef SSD_POSTPROCESS_H
#define SSD_POSTPROCESS_H

#include <stdint.h>
#include "app_config.h"

typedef struct {
  float x1;
  float y1;
  float x2;
  float y2;
  float score;
} ssd_det_t;

typedef struct {
  int x;
  int y;
  int count;
} grid_point_t;

void ssd_post_init(void);

uint32_t ssd_post_process(
  const int8_t *scores_q,
  const int8_t *boxes_q,
  uint32_t frame_w,
  uint32_t frame_h,
  ssd_det_t *out_dets,
  uint32_t out_max
);

uint32_t ssd_build_grid_from_dets(
  const ssd_det_t *dets,
  uint32_t det_count,
  grid_point_t *grid,
  uint32_t grid_max
);

#endif // SSD_POSTPROCESS_H
