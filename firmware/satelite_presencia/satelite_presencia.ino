/*
 * SATELITE DE PRESENCIA para A.R.I.A — ESP32 solo, sin hardware extra
 * ===================================================================
 * Escanea Bluetooth BLE buscando el celular/reloj del jefe y le avisa al PC:
 * A.R.I.A sabe si estás en casa y te saluda cuando llegas ("Bienvenido jefe").
 *
 * Protocolo /ws/presencia (ver app/web/presencia_ws.py):
 *   {"type":"presencia","presente":true,"rssi":-62}  = lo detecté
 *   {"type":"presencia","presente":false}            = ya no está
 *   servidor responde: {"type":"ok","presente":true}
 *
 * Librerías (Arduino IDE → Gestor de librerías):
 *   - WebSockets  (Markus Sattler)
 *   - ArduinoJson (Benoit Blanchon)
 *   (El escaneo BLE usa la librería que YA viene con el soporte ESP32.)
 *
 * LED integrado:
 *   latido            = buscando red/servidor
 *   parpadeo lento    = escaneando, jefe FUERA
 *   fijo              = escaneando, jefe EN CASA
 *   doble destello    = acaba de reportar un cambio
 */

#include <ArduinoJson.h>
#include <BLEDevice.h>
#include <BLEScan.h>
#include <WebSocketsClient.h>
#include <WiFi.h>

// ---------- EDITA ESTO ----------
const char* WIFI_SSID   = "Rochy72";
const char* WIFI_PASS   = "1139426546";
const char* ARIA_HOST   = "192.168.1.25";  // IP fija o IP Tailscale del PC
const int   ARIA_PORT   = 8000;
const char* ARIA_TOKEN  = "afe27c8940d6ee04311ff0b2a525e336";              // ACCESS_TOKEN del .env (si usas uno)

// A quién buscar. Deja "" y usa el modo DESCUBRIR (abajo) para ver las MAC.
// ¡Importante!: si tu celular rota su dirección BLE aleatoria, busca por
// NOMBRE (el del Bluetooth de tu teléfono) o usa la MAC de un smartwatch.
const char* OBJETIVOS_MAC[] = {
  "D4:5B:51:4A:44:21",   // celular de Samuel
};
const int NUM_MACS = sizeof(OBJETIVOS_MAC) / sizeof(OBJETIVOS_MAC[0]);

const char* OBJETIVOS_NOMBRE[] = {
  // "Galaxy S23",          // nombre visible del Bluetooth
};
const int NUM_NOMBRES = sizeof(OBJETIVOS_NOMBRE) / sizeof(OBJETIVOS_NOMBRE[0]);

// PON true, sube el firmware, abre el Serial Monitor (115200) 60 s y copia
// la MAC/nombre de tu celular en las listas de arriba. Vuelve a poner false.
const bool MODO_DESCUBRIR = false;
// --------------------------------

const unsigned long INTERVALO_SCAN_MS = 45000;  // un escaneo cada 45 s
const int SEGUNDOS_SCAN = 3;
const int FALLOS_PARA_SALIR = 3;  // 3 escaneos sin verte (~2 min) = saliste

const int PIN_LED = 2;

WebSocketsClient ws;
BLEScan* escaner;

bool presente = false;      // último estado confirmado (con histéresis)
int fallosSeguidos = 0;
unsigned long ultimoScan = 0;

enum Estado { CONECTANDO, ACTIVO };
Estado estado = CONECTANDO;

void led(bool on) { digitalWrite(PIN_LED, on ? HIGH : LOW); }

void parpadeos(int veces, int ms = 120) {
  for (int i = 0; i < veces; i++) { led(true); delay(ms); led(false); delay(ms); }
}

void enviarPresencia(bool esta) {
  JsonDocument doc;
  doc["type"] = "presencia";
  doc["presente"] = esta;
  char buf[96];
  serializeJson(doc, buf);
  ws.sendTXT(buf);
}

void eventoWS(WStype_t tipo, uint8_t* datos, size_t largo) {
  switch (tipo) {
    case WStype_CONNECTED:
      Serial.println("[ARIA] Conectado al servidor");
      estado = ACTIVO;
      ultimoScan = 0;              // escanear de inmediato
      enviarPresencia(presente);   // sincronizar el estado actual
      break;
    case WStype_DISCONNECTED:
      estado = CONECTANDO;
      Serial.println("[ARIA] Sin conexión con el servidor");
      break;
    case WStype_TEXT: {
      JsonDocument doc;
      if (deserializeJson(doc, datos, largo)) return;
      if (strcmp(doc["type"] | "", "ok") == 0) parpadeos(2, 60);  // cambio recibido
      return;
    }
    default:
      break;
  }
}

void conectarWiFi() {
  Serial.printf("[ARIA] Conectando a %s…\n", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) { delay(300); parpadeos(1, 60); }
  Serial.printf("[ARIA] WiFi OK, mi IP: %s\n", WiFi.localIP().toString().c_str());
}

void conectarServidor() {
  String ruta = "/ws/presencia";
  if (String(ARIA_TOKEN).length()) ruta += String("?token=") + ARIA_TOKEN;
  ws.begin(ARIA_HOST, ARIA_PORT, ruta.c_str());
  ws.onEvent(eventoWS);
  ws.setReconnectInterval(5000);
}

bool esObjetivo(const String& mac, const char* nombre) {
  for (int i = 0; i < NUM_MACS; i++) {
    String objetivo = OBJETIVOS_MAC[i];
    objetivo.toUpperCase();
    if (objetivo.length() && objetivo == mac) return true;
  }
  for (int i = 0; i < NUM_NOMBRES; i++) {
    String objetivo = OBJETIVOS_NOMBRE[i];
    if (objetivo.length() && nombre && objetivo.equalsIgnoreCase(nombre)) return true;
  }
  return false;
}

void listarDispositivos(BLEScanResults* resultados) {
  Serial.println("[ARIA] DISPOSITIVOS BLE VISIBLES (copia MAC o nombre):");
  for (int i = 0; i < resultados->getCount(); i++) {
    BLEAdvertisedDevice d = resultados->getDevice(i);
    Serial.printf("  %s | rssi %d | nombre: %s\n",
                  d.getAddress().toString().c_str(),
                  d.getRSSI(),
                  d.haveName() ? d.getName().c_str() : "(sin nombre)");
  }
  Serial.println("[ARIA] fin de la lista");
}

void escanear() {
  BLEScanResults* resultados = escaner->start(SEGUNDOS_SCAN, false);
  bool visto = false;
  int mejorRssi = -127;

  if (MODO_DESCUBRIR) listarDispositivos(resultados);

  for (int i = 0; i < resultados->getCount(); i++) {
    BLEAdvertisedDevice d = resultados->getDevice(i);
    String mac = String(d.getAddress().toString().c_str());
    mac.toUpperCase();
    const char* nombre = d.haveName() ? d.getName().c_str() : nullptr;
    if (esObjetivo(mac, nombre)) {
      visto = true;
      mejorRssi = max(mejorRssi, d.getRSSI());
    }
  }
  escaner->clearResults();

  if (visto) {
    fallosSeguidos = 0;
    if (!presente) {
      presente = true;
      Serial.printf("[ARIA] El jefe LLEGÓ a casa (rssi %d)\n", mejorRssi);
      if (estado == ACTIVO) enviarPresencia(true);
    }
  } else if (presente || fallosSeguidos > 0) {
    fallosSeguidos++;
    if (presente && fallosSeguidos >= FALLOS_PARA_SALIR) {
      presente = false;
      Serial.println("[ARIA] El jefe SALIÓ de casa");
      if (estado == ACTIVO) enviarPresencia(false);
    }
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(PIN_LED, OUTPUT);

  BLEDevice::init("");
  escaner = BLEDevice::getScan();
  escaner->setActiveScan(true);   // pide nombres, no solo MAC
  escaner->setInterval(49);
  escaner->setWindow(49);

  conectarWiFi();
  conectarServidor();
}

void loop() {
  ws.loop();

  if (estado == CONECTANDO) {               // latido mientras reconecta
    static unsigned long ultimoLatido = 0;
    if (millis() - ultimoLatido > 700) { ultimoLatido = millis(); led(!digitalRead(PIN_LED)); }
    return;
  }

  // LED refleja el estado: fijo = en casa, parpadeo lento = fuera
  static unsigned long ultimoParpadeo = 0;
  if (!presente && millis() - ultimoParpadeo > 900) {
    ultimoParpadeo = millis();
    led(!digitalRead(PIN_LED));
  } else if (presente) {
    led(true);
  }

  if (millis() - ultimoScan >= INTERVALO_SCAN_MS) {
    ultimoScan = millis();
    escanear();
  }
}
