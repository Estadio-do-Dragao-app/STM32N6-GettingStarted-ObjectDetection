#ifndef MQTT_PUBLISHER_H
#define MQTT_PUBLISHER_H

#include <stdint.h>
#include <stdbool.h>
#include "ssd_postprocess.h"

bool mqtt_pub_init(void);

bool mqtt_pub_crowd_density(
  uint32_t total_people,
  const grid_point_t *grid,
  uint32_t grid_count,
  uint32_t inference_ms,
  float avg_confidence
);

#endif /* MQTT_PUBLISHER_H */
