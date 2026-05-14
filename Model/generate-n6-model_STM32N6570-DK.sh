#!/bin/bash

set -eu # Exit on any error, Exit on unset variable

echo "ERROR: This project is currently configured for SSDLite on STM32N6570-DK."
echo "Running this script would regenerate artifacts for YOLOX and break SSDLite post-processing."
echo "Do not run this script unless you are intentionally migrating the full pipeline to YOLOX."
echo "Use the committed files under Model/STM32N6570-DK (network.c, stai_network.c, network_data.hex)."
exit 1
