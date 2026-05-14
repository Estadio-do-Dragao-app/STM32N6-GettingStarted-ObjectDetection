#include "mqtt_publisher.h"
#include "app_config.h"

#include <stdio.h>
#include <string.h>

#if defined(USE_LWIP_MQTT)
#include "lwip/apps/mqtt.h"
#include "lwip/ip_addr.h"

static mqtt_client_t *g_client = NULL;
static bool g_connected = false;

static void mqtt_request_cb(void *arg, err_t err)
{
  (void)arg;
  (void)err;
}

static void mqtt_connection_cb(mqtt_client_t *client, void *arg, mqtt_connection_status_t status)
{
  (void)client;
  (void)arg;
  g_connected = (status == MQTT_CONNECT_ACCEPTED);
}

bool mqtt_pub_init(void)
{
  ip_addr_t broker_ip;
  struct mqtt_connect_client_info_t ci;

  g_client = mqtt_client_new();
  if (!g_client) return false;

  memset(&ci, 0, sizeof(ci));
  ci.client_id = APP_CAMERA_ID;
  ci.keep_alive = 30;

  if (!ipaddr_aton(APP_MQTT_BROKER_IP, &broker_ip)) return false;

  mqtt_client_connect(g_client, &broker_ip, APP_MQTT_BROKER_PORT, mqtt_connection_cb, NULL, &ci);
  return true;
}

static bool mqtt_publish_payload(const char *payload)
{
  if (!g_connected || !g_client || !payload) return false;
  err_t rc = mqtt_publish(g_client, APP_MQTT_TOPIC_CONGEST, payload,
                          (u16_t)strlen(payload), 0, 0, mqtt_request_cb, NULL);
  return rc == ERR_OK;
}

#else
/* UART fallback: JSON is printed to serial and bridged to MQTT by the host */

bool mqtt_pub_init(void)
{
  return true;
}

static bool mqtt_publish_payload(const char *payload)
{
  if (!payload) return false;
  /* Host script reads lines starting with MQTT_JSON: and publishes to broker */
  printf("MQTT_JSON:%s\r\n", payload);
  return true;
}

#endif /* USE_LWIP_MQTT */

bool mqtt_pub_crowd_density(
  uint32_t total_people,
  const grid_point_t *grid,
  uint32_t grid_count,
  uint32_t inference_ms,
  float avg_confidence)
{
  char payload[APP_JSON_MAX_LEN];
  int used = 0;

  used += snprintf(payload + used, sizeof(payload) - (size_t)used,
    "{\"event_type\":\"crowd_density\",\"level\":%d,\"grid_data\":[", APP_LEVEL);

  for (uint32_t i = 0; i < grid_count; i++) {
    used += snprintf(payload + used, sizeof(payload) - (size_t)used,
      "%s{\"x\":%d,\"y\":%d,\"count\":%d}",
      (i == 0U) ? "" : ",", grid[i].x, grid[i].y, grid[i].count);
    if ((size_t)used >= sizeof(payload)) return false;
  }

  used += snprintf(payload + used, sizeof(payload) - (size_t)used,
    "],\"total_people\":%lu,\"metadata\":{\"camera_id\":\"%s\","
    "\"coordinate_unit\":\"%s\",\"wait_time_sec\":%d,"
    "\"inference_ms\":%lu,\"avg_confidence\":%.3f}}",
    (unsigned long)total_people,
    APP_CAMERA_ID, APP_COORDINATE_UNIT, APP_WAIT_TIME_SEC,
    (unsigned long)inference_ms, (double)avg_confidence);

  if ((size_t)used >= sizeof(payload)) return false;

  return mqtt_publish_payload(payload);
}
