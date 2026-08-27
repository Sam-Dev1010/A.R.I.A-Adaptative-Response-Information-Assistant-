#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include "model_data.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// --- Configuración de Red e Integración con A.R.I.A ---
const char* WIFI_SSID = "Rochy72";
const char* WIFI_PASS = "1139426546";
const char* ARIA_ALERT_URL = "http://192.168.1.25:8000/api/esp32/event"; // Ajusta a la IP de tu PC

// --- Asignación de RAM Estática (64 KB para el ESP32 clásico) ---
constexpr int kTensorArenaSize = 64 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

// --- Punteros Globales de TensorFlow Lite ---
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Función para notificar a la API de A.R.I.A
void notifyARIA(float score) {
  if (WiFi.status() != WL_CONNECTED) return;

  HTTPClient http;
  http.begin(ARIA_ALERT_URL);
  http.addHeader("Content-Type", "application/json");

  String payload = "{\"node_id\": \"esp32_ia_local\", \"prediction\": " + String(score, 4) + "}";
  int httpResponseCode = http.POST(payload);

  if (httpResponseCode > 0) {
    Serial.printf("[A.R.I.A] Evento enviado. Respuesta HTTP: %d\n", httpResponseCode);
  } else {
    Serial.printf("[A.R.I.A] Error enviando alerta: %s\n", http.errorToString(httpResponseCode).c_str());
  }
  http.end();
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n>>> Iniciando Nodo de IA Local A.R.I.A (ESP32) <<<");

  // Conexión Wi-Fi
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("Conectando a Wi-Fi...");
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\nWi-Fi conectado. IP: %s\n", WiFi.localIP().toString().c_str());
  } else {
    Serial.println("\nContinuando en modo fuera de línea...");
  }

  // Inicialización de TensorFlow Lite Micro
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter, "Error de versión del esquema TFLite.");
    return;
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Fallo al asignar la memoria Tensor Arena.");
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);
  Serial.println("[OK] Sistema de IA embebida e intérprete inicializados.");
}

void loop() {
  if (!input) return;

  // --- 1. Lectura / Simulación de datos ---
  // Cuando tengas tus sensores, asignas los datos normalizados a input->data.f[]
  input->data.f[0] = 0.85f; 

  // --- 2. Ejecución de Inferencia Local ---
  unsigned long start_time = millis();
  TfLiteStatus invoke_status = interpreter->Invoke();
  unsigned long latency = millis() - start_time;

  if (invoke_status == kTfLiteOk) {
    float prediction = output->data.f[0];
    Serial.printf("Inferencia: %.4f | Latencia: %lu ms\n", prediction, latency);

    // --- 3. Umbral de Activación ---
    if (prediction > 0.80f) {
      Serial.println("¡Detección positiva! Alertando a A.R.I.A...");
      notifyARIA(prediction);
    }
  } else {
    Serial.println("Error ejecutando la inferencia.");
  }

  delay(3000);
}