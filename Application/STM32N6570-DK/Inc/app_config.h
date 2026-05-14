 /**
 ******************************************************************************
 * @file    app_config.h
 * @author  GPM Application Team
 *
 ******************************************************************************
 * @attention
 *
 * Copyright (c) 2023 STMicroelectronics.
 * All rights reserved.
 *
 * This software is licensed under terms that can be found in the LICENSE file
 * in the root directory of this software component.
 * If no LICENSE file comes with this software, it is provided AS-IS.
 *
 ******************************************************************************
 */

#ifndef APP_CONFIG
#define APP_CONFIG

#include "arm_math.h"

#define USE_DCACHE

/*Defines: CMW_MIRRORFLIP_NONE; CMW_MIRRORFLIP_FLIP; CMW_MIRRORFLIP_MIRROR; CMW_MIRRORFLIP_FLIP_MIRROR;*/
#define CAMERA_FLIP CMW_MIRRORFLIP_NONE

#define ASPECT_RATIO_CROP       (1) /* Crop both pipes to nn input aspect ratio; Original aspect ratio kept */
#define ASPECT_RATIO_FIT        (2) /* Resize both pipe to NN input aspect ratio; Original aspect ratio not kept */
#define ASPECT_RATIO_FULLSCREEN (3) /* Resize camera image to NN input size and display a maximized image. See Doc/Build-Options.md#aspect-ratio-mode */
#define ASPECT_RATIO_MODE ASPECT_RATIO_CROP

/* Color mode */
#define COLOR_BGR (0)
#define COLOR_RGB (1)
#define COLOR_MODE    COLOR_RGB

/* Classes */
#define NB_CLASSES   (1)
#define CLASSES_TABLE const char* classes_table[NB_CLASSES] = {"person"}

/* SSD post-processing */
#define APP_SSDLITE_PRIOR_COUNT  3000
#define APP_SSDLITE_CONF_THRESH  0.20f
#define APP_SSDLITE_NMS_IOU      0.40f
#define APP_SSDLITE_MAX_DETS     64
#define APP_GRID_MAX_POINTS      64

/* MQTT / UART-bridge config */
#define APP_CAMERA_ID          "CAM_REAL_001"
#define APP_LEVEL              0
#define APP_COORDINATE_UNIT    "pixels"
#define APP_WAIT_TIME_SEC      0
#define APP_MQTT_BROKER_IP     "192.168.1.10"
#define APP_MQTT_BROKER_PORT   1883
#define APP_MQTT_TOPIC_CONGEST "stadium/events/congestion"
#define APP_JSON_MAX_LEN       2048

/* Display */
#define WELCOME_MSG_1         "SSDLite MobileNetV3-Small 300x300 int8"
#define WELCOME_MSG_2         "Person Detection + MQTT"

#endif
