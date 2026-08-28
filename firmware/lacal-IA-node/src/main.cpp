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

// --- Pin del LED indicador (GPIO 2 en la placa ESP32 devkit) ---
const int LED_PIN = 2;

// --- Umbral de activación de la alerta ---
constexpr float kUmbralAlerta = 0.80f;

// --- Asignación de RAM Estática (64 KB para el ESP32 clásico) ---
constexpr int kTensorArenaSize = 64 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

// --- Punteros Globales de TensorFlow Lite ---
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

bool is_initialized = false;

// ------------------------- Utilidades ------------------------------------

bool isValidTFLiteModel() {
  // La firma FlatBuffers de TensorFlow Lite es "TFL3" en la posición 4.
  if (g_model_len < 32) return false;
  return (g_model[4] == 'T' && g_model[5] == 'F' &&
          g_model[6] == 'L' && g_model[7] == '3');
}

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

void avisarDeteccion(float prediction) {
  // LED: parpadea 3 veces para señalar la detección y queda encendido.
  for (int i = 0; i < 3; i++) {
    digitalWrite(LED_PIN, HIGH);
    delay(120);
    digitalWrite(LED_PIN, LOW);
    delay(120);
  }
  digitalWrite(LED_PIN, HIGH);

  Serial.println("¡Detección positiva! Alertando a A.R.I.A...");
  notifyARIA(prediction);
}

// ------------------------- Setup -----------------------------------------

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n>>> Iniciando Nodo de IA Local A.R.I.A (ESP32) <<<");

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW); // Comienza apagado

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

  // El header debe contener un modelo real: se genera con scripts/export_tflite.py
  if (!isValidTFLiteModel()) {
    Serial.println("[!] 'g_model' en model_data.h no es una red válida aún. ESP32 en espera...");
    return;
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
  is_initialized = true;

  // LED fijo encendido = IA embebida 100% cargada y lista.
  digitalWrite(LED_PIN, HIGH);
  Serial.println("[OK] Sistema de IA embebida e intérprete inicializados.");
}

// ------------------------- Loop ------------------------------------------

void loop() {
  if (!is_initialized) {
    // Heartbeat: parpadeo lento mientras se espera un modelo exportado válido.
    digitalWrite(LED_PIN, HIGH);
    delay(200);
    digitalWrite(LED_PIN, LOW);
    delay(2000);
    return;
  }

  if (input == nullptr) return;

  // --- 1. Lectura / Normalización de sensores --------------------------------
  // TODO(sensores): mapear las lecturas reales (ADC, IMU, pulsos, sonido...)
  // al rango que espera la red y copiarlas en input->data.f[].
  //   Ejemplo con un divisor de tensión en GPIO34 (0..4095):
  //     float v = (float)analogRead(34) / 4095.0f;
  //     input->data.f[0] = (v - kMinLectura) / (kMaxLectura - kMinLectura);
  // La forma del tensor (¿1 valor, N muestras, imágenes?) la define el modelo
  // exportado; consulta input->dims y rellena cada dimensión.
  // Por ahora se deja un valor de prueba normalizado (0..1):
  input->data.f[0] = 0.85f;

  // --- 2. Ejecución de Inferencia Local -------------------------------------
  unsigned long start_time = millis();
  TfLiteStatus invoke_status = interpreter->Invoke();
  unsigned long latency = millis() - start_time;

  if (invoke_status == kTfLiteOk) {
    float prediction = output->data.f[0];
    Serial.printf("Inferencia: %.4f | Latencia: %lu ms\n", prediction, latency);

    // --- 3. Umbral de Activación ---------------------------------------------
    if (prediction > kUmbralAlerta) {
      avisarDeteccion(prediction);
    }
  } else {
    Serial.println("Error ejecutando la inferencia.");
  }

  delay(3000);
}