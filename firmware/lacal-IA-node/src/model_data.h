#ifndef MODEL_DATA_H_
#define MODEL_DATA_H_

#include <Arduino.h>

// Placeholder temporal hasta exportar la red neuronal entrenada
const unsigned char g_model[] PROGMEM = {
  0x1c, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33, 0x00, 0x00, 0x00, 0x00
};
const int g_model_len = sizeof(g_model);

#endif // MODEL_DATA_H_