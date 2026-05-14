# STM32N6570-DK SSDLite - Team Handoff (Build + Flash + Camera)

Este guia deixa o projeto pronto para qualquer colega fazer build e flash no STM32CubeProgrammer sem surpresas.

## 1) Estado esperado do projeto

- Board: `STM32N6570-DK`
- Modelo ativo: `ssdlite_mobilenetv3small_pt_coco_person_300_qdq_int8_OE_3_3_1`
- Post-processamento: `ssd_postprocess.c` (SSD custom)

Importante:
- Nao correr `Model/generate-n6-model_STM32N6570-DK.sh` se o objetivo for manter SSDLite.
- Esse script gera configuracao para outro modelo (YOLOX) e pode levar a 0 deteccoes.

## 2) Build (firmware da aplicacao)

Na pasta:

```bash
cd Application/STM32N6570-DK
make clean
make -j8
make sign
```

Isto gera:
- `build/Application/STM32N6570-DK/Project.bin`
- `build/Application/STM32N6570-DK/Project_sign.hex`

## 3) Flash no STM32CubeProgrammer (ordem obrigatoria)

1. Colocar board em `Development mode`.
2. Abrir STM32CubeProgrammer e conectar por ST-LINK (`Hot plug`).
3. Programar, por esta ordem:
   1. `FSBL/ai_fsbl.hex`
   2. `Application/STM32N6570-DK/build/Application/STM32N6570-DK/Project_sign.hex`
   3. `Model/STM32N6570-DK/network_data.hex`
4. Colocar board em `Boot from flash mode`.
5. Fazer power cycle.

Notas:
- Se mudar o modelo, reprogramar sempre `network_data.hex`.
- Se mudar so codigo C, normalmente basta reprogramar `Project_sign.hex`.

## 4) Verificacao rapida na consola UART

Ao arrancar, validar:
- `NN model: ssdlite_mobilenetv3small_pt_coco_person_300_qdq_int8_OE_3_3_1`

Se aparecer outro nome de modelo, ha mismatch de firmware/modelo.

## 5) Metricas disponiveis (runtime)

A cada 30 frames, a UART imprime:

```text
METRICS:{"frame":...,"det_count":...,"inf_ms":...,"fps":...,"ema_inf_ms":...,"ema_fps":...,"avg_conf":...}
```

Significado:
- `det_count`: pessoas detetadas no frame
- `inf_ms`: latencia de inferencia desse frame (ms)
- `fps`: FPS instantaneo aproximado (`1000/inf_ms`)
- `ema_inf_ms`: media movel de latencia
- `ema_fps`: media movel de FPS
- `avg_conf`: confianca media das deteccoes do frame (0 a 1)

No JSON `MQTT_JSON` agora tambem seguem:
- `metadata.inference_ms`
- `metadata.avg_confidence`

## 6) Qualidade do modelo (o que estas metricas medem)

Estas metricas medem desempenho em runtime, nao precisao de deteccao.

Para medir qualidade real de modelo (precision/recall/mAP), e preciso:
- conjunto de imagens/videos com anotacoes (ground truth)
- avaliacao offline no PC com o mesmo threshold usado na board

Como aproximacao rapida em campo:
- observar `det_count` e `avg_conf` em cenarios controlados (0, 1, 2+ pessoas)
- validar estabilidade de `det_count` frame a frame (evitar oscilacoes grandes sem motivo)

Para avaliar com labels YOLO (ex.: coco128), usar:
- `tools/eval_person_yolo.py`
