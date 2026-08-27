/*
 * SATELITE DE VOZ para SIA — ESP32 WROOM + INMP441 + MAX98357A
 * ============================================================
 * Mantén el botón: el micrófono manda tu voz al PC de Samuel.
 * Suéltalo:      SIA piensa y responde HABLANDO por la bocina.
 *
 * Protocolo /ws/satelite (ver app/web/satelite_ws.py):
 *   - BINARIO hacia el PC  = PCM crudo 16 kHz mono 16-bit (micrófono)
 *   - {"type":"fin"}       = soltó el botón
 *   - BINARIO desde el PC  = PCM crudo 16 kHz mono 16-bit (bocina)
 *   - {"type":"audio_end"} = terminó la respuesta
 *
 * Librerías (Arduino IDE → Gestor de librerías):
 *   - WebSockets  (Markus Sattler)
 *   - ArduinoJson (Benoit Blanchon)
 *
 * Cableado:
 *   INMP441    VDD=3.3V  GND=GND  L/R=GND  SCK=D14  WS=D15  SD=D32
 *   MAX98357A  VIN=5V*   GND=GND  BCLK=D26 LRC=D25  DIN=D22 (*o 3.3V)
 *   BOTÓN      un pin a D4, otro a GND (usa INPUT_PULLUP, sin resistencia)
 */

#include <ArduinoJson.h>
#include <ESP_I2S.h>
#include <WiFi.h>
#include <WebSocketsClient.h>

// ---------- EDITA ESTO ----------
const char* WIFI_SSID   = "TU_WIFI";
const char* WIFI_PASS   = "TU_CLAVE";
const char* SIA_HOST    = "192.168.1.25";  // IP fija o IP Tailscale del PC
const int   SIA_PORT    = 8000;
const char* SIA_TOKEN   = "";              // ACCESS_TOKEN del .env (si usas uno)
// --------------------------------

const int PIN_BOTON = 4;
const int PIN_LED   = 2;

const int SR          = 16000;             // Hz, igual que el servidor
const int BYTES_BLOQUE = 512;              // por lectura/envío del mic

I2SClass i2s;
WebSocketsClient ws;

enum Estado { CONECTANDO, LISTO, GRABANDO, PENSANDO, HABLANDO };
Estado estado = CONECTANDO;
bool botonAntes = false;

void led(bool on) { digitalWrite(PIN_LED, on ? HIGH : LOW); }

void parpadeos(int veces, int ms = 120) {
  for (int i = 0; i < veces; i++) { led(true); delay(ms); led(false); delay(ms); }
}

void enviar(const char* tipo) {
  StaticJsonDocument<64> doc;
  doc["type"] = tipo;
  char buf[64];
  serializeJson(doc, buf);
  ws.sendTXT(buf);
}

void eventoWS(WStype_t tipo, uint8_t* datos, size_t largo) {
  switch (tipo) {
    case WStype_CONNECTED:
      Serial.println("[SIA] Conectado al PC");
      break;
    case WStype_DISCONNECTED:
      estado = CONECTANDO;
      Serial.println("[SIA] Sin conexión con el PC");
      break;
    case WStype_BIN:  // voz de SIA → directo a la bocina
      estado = HABLANDO;
      led(true);
      i2s.write(datos, largo);
      return;
    case WStype_TEXT: {
      JsonDocument doc;
      if (deserializeJson(doc, datos, largo)) return;
      const char* t = doc["type"] | "";
      if (strcmp(t, "audio_end") == 0) {
        estado = LISTO;
        led(false);
      } else if (strcmp(t, "estado") == 0) {
        const char* v = doc["valor"] | "";
        if (strcmp(v, "hablando") == 0) { estado = HABLANDO; }
        else if (strcmp(v, "pensando") == 0) { estado = PENSANDO; }
      } else if (strcmp(t, "error") == 0) {
        Serial.printf("[SIA] Error: %s\n", doc["message"] | "?");
        parpadeos(3);
        estado = LISTO;
      }
      return;
    }
    default:
      break;
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(PIN_BOTON, INPUT_PULLUP);
  pinMode(PIN_LED, OUTPUT);

  // Un solo bus I2S: micrófono (DIN) y bocina (DOUT).
  i2s.setPins(14, 15, 22, 32, -1);  // bclk, ws, dout, din, mclk
  if (!i2s.begin(I2S_MODE_STD, SR, I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO)) {
    Serial.println("[SIA] No pude iniciar el I2S");
    while (true) { parpadeos(5); delay(1000); }
  }

  Serial.printf("[SIA] Conectando a %s…\n", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) { delay(300); parpadeos(1, 60); }
  Serial.printf("[SIA] WiFi OK, mi IP: %s\n", WiFi.localIP().toString().c_str());

  String ruta = "/ws/satelite";
  if (String(SIA_TOKEN).length()) ruta += String("?token=") + SIA_TOKEN;
  ws.begin(SIA_HOST, SIA_PORT, ruta.c_str());
  ws.onEvent(eventoWS);
  ws.setReconnectInterval(5000);
}

void loop() {
  ws.loop();

  bool presionado = digitalRead(PIN_BOTON) == LOW;
  if (presionado != botonAntes) {           // flanco del botón
    botonAntes = presionado;
    if (presionado && estado != CONECTANDO && estado != HABLANDO) {
      estado = GRABANDO;
      led(true);
    } else if (!presionado && estado == GRABANDO) {
      enviar("fin");                        // suelto: que SIA responda
      estado = PENSANDO;
    }
  }

  if (estado == GRABANDO) {                 // micrófono → PC
    uint8_t bloque[BYTES_BLOQUE];
    size_t leidos = i2s.readBytes(bloque, BYTES_BLOQUE);
    if (leidos > 0) ws.sendBIN(bloque, leidos);
  }

  if (estado == CONECTANDO) {               // latido mientras reconecta
    static unsigned long ultimo = 0;
    if (millis() - ultimo > 700) { ultimo = millis(); led(!digitalRead(PIN_LED)); }
  }
}
