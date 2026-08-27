# A.R.I.A — Asistente Personal de IA

**A.R.I.A** (*Adaptive Response & Information Assistant*) es una asistente personal de escritorio inspirada en JARVIS (Iron Man). Escucha por voz,
responde hablando, usa un LLM como cerebro, ejecuta herramientas controladas sobre
el sistema y recuerda tus datos entre sesiones.

> Antes llamada **SIA** — mismo proyecto, nueva identidad.
> Proyecto completo: todas las fases del roadmap están implementadas ✅

## Arquitectura

```
sia/
├── app/
│   ├── core/          # Núcleo: configuración y logging
│   │   ├── config.py  # Settings con pydantic-settings (variables de entorno / .env)
│   │   └── logging.py # Logging estructurado (campos JSON, sin secretos)
│   ├── ai/            # Capa LLM
│   │   ├── schemas.py # ChatMessage, ChatResponse, TokenUsage, ToolCall
│   │   ├── orchestrator.py  # AssistantOrchestrator: historial + tools + memoria
│   │   ├── factory.py       # Construye todos los componentes desde Settings
│   │   └── providers/
│   │       ├── base.py                 # Contrato LLMProvider + LLMError
│   │       └── openai_compatible.py    # OpenAI, Groq, OpenRouter, Ollama, LM Studio...
│   ├── voice/         # Voz
│   │   ├── base.py    # Contratos STTProvider / TTSProvider + VoiceError
│   │   ├── stt.py     # GoogleSTTProvider (micrófono con sounddevice)
│   │   ├── tts.py     # EdgeTTSProvider (voces neuronales + reproductor)
│   │   └── assistant.py  # VoiceAssistant: ciclo completo voz ↔ LLM ↔ voz
│   ├── memory/        # Memoria persistente
│   │   └── manager.py # MemoryManager: SQLite (sesiones, mensajes, hechos)
│   ├── tools/         # Sistema de herramientas con permisos
│   │   ├── base.py    # BaseTool (name, description, JSON Schema, permiso)
│   │   ├── policy.py  # ToolPolicy + PermissionDenied (SAFE/CONFIRM/RESTRICTED)
│   │   ├── registry.py# ToolRegistry (registro único por nombre)
│   │   ├── builtins.py# get_time, get_system_info, open_website
│   │   ├── memory_tools.py  # remember, forget
│   │   ├── network_tools.py # get_weather, search_wikipedia (sin API key)
│   │   └── schemas.py # ToolSpec, FunctionSpec, ToolResult
│   ├── web/           # GUI web (FASE 8)
│   │   ├── ws.py      # ChatConnection: WebSocket + confirmaciones de tools
│   │   ├── interface_ws.py  # Interfaz de voz: audio en streaming, voz primero
│   │   ├── satelite_ws.py   # Satélite ESP32: push-to-talk por habitaciones
│   │   └── static/    # index.html (chat) + interface.html (AI Core por voz/texto)
│   └── main.py        # FastAPI: /health, / (GUI), /ws/chat
├── tests/             # Tests unitarios (pytest)
├── scripts/
│   ├── chat.py        # REPL de chat por terminal
│   ├── voice.py       # Sesión completa por voz (JARVIS mode)
│   ├── sia_app.py     # Ventana nativa de escritorio (PyQt6)
│   ├── sia_movil.py   # Servidor HTTPS para usar SIA desde el celular
│   └── sincronizar_movil.py  # Interfaz web → mobile/www (app Android)
├── firmware/          # Satélite de voz ESP32 (por habitación)
├── data/              # Datos locales (sia.db, audio TTS)
├── pyproject.toml     # Empaquetado + ruff + pytest
├── requirements.txt   # Dependencias base
└── requirements-voice.txt  # Dependencias de voz
```

Flujo del asistente:

```
Voz → STT → LLM → ¿Tool? → ToolPolicy → ejecución → resultado → LLM → respuesta → TTS
```

## Instalación en cualquier PC

### 1. Requisitos previos

- **Python 3.12+** (`python3 --version`)
- **ffmpeg** (convierte el audio del navegador) — recomendado
- **Un reproductor de audio**: `mpv`, `ffplay`, `aplay` o `paplay` (para que SIA hable)
- **Una API key de LLM**. La opción gratis: [Groq](https://console.groq.com/keys)
- Para la voz: micrófono y altavoces

#### Linux (Fedora)

```bash
sudo dnf install python3 python3-pip ffmpeg mpv git
```

#### Linux (Debian/Ubuntu)

```bash
sudo apt install python3 python3-venv python3-pip ffmpeg mpv git
```

#### Windows

1. Instala [Python 3.12+](https://www.python.org/downloads/) marcando "Add to PATH".
2. Instala [ffmpeg](https://www.gyan.dev/ffmpeg/builds/) y agrégalo al PATH.
3. Opcional: [mpv](https://mpv.io/installation/).

### 2. Copiar el proyecto e instalar dependencias

Copia la carpeta del proyecto a la PC nueva (pendrive, `git clone` o zip), luego:

```bash
cd sia
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt      # base + herramientas de desarrollo
pip install -r requirements-voice.txt    # voz (STT + TTS neuronal)
pip install -r requirements-desktop.txt  # solo si quieres la app nativa PyQt
```

### 3. Configurar `.env`

```bash
cp .env.example .env
nano .env   # o cualquier editor
```

Lo mínimo para funcionar:

```ini
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_API_KEY=gsk_...            # tu key de https://console.groq.com/keys
LLM_MODEL=openai/gpt-oss-20b
```

Opcional — voz natural (por defecto ya viene configurada así):

```ini
TTS_PROVIDER=auto              # edge = voz neuronal humana
TTS_VOICE=es-MX-DaliaNeural    # mujer mexicana; es-MX-JorgeNeural = hombre
TTS_RATE=+3%
TTS_PITCH=+0Hz
```

> Sin internet puedes usar `TTS_PROVIDER=piper` (síntesis local instantánea,
> menos natural) y un LLM local con Ollama o LM Studio como `LLM_BASE_URL`.

### 4. Ejecutar A.R.I.A

```bash
# Interfaz futurista por voz en el navegador
uvicorn app.main:app --host 0.0.0.0 --port 8000
# → abre http://localhost:8000

# App de escritorio nativa (requiere requirements-desktop.txt)
python scripts/sia_app.py

# Chat por terminal
python scripts/chat.py

# Solo voz desde la terminal (modo JARVIS)
python scripts/voice.py
```

Verifica que todo responde: `curl http://localhost:8000/health`

## Docker: backend en contenedor

Empaqueta el servidor completo en una imagen ligera y portátil: se mueve a
cualquier servidor con un `docker compose up` y la memoria (SQLite) queda en
la carpeta `data/` del host, fuera del contenedor.

Requisitos: Docker (y docker compose plugin). El `.env` de siempre provee la
configuración; los secretos nunca entran en la imagen.

```bash
# Construir y levantar (primera vez o tras cambios)
docker compose up -d --build

# Ver logs en vivo
docker compose logs -f

# Actualizar tras cambiar código
git pull && docker compose up -d --build

# Apagar
docker compose down          # la memoria persiste en ./data
```

- Web: `http://localhost:8000` · Salud: `http://localhost:8000/health`
- Para moverla a otro servidor: copia el proyecto (o solo `Dockerfile`,
  `requirements*.txt`, `docker-compose.yml`, `app/` y tu `.env`) y ejecuta
  `docker compose up -d --build`. Si también copias `data/`, viaja con toda
  su memoria.
- En producción detrás de un proxy define `ACCESS_TOKEN` en `.env`: dentro
  del contenedor nada es "localhost", así que todos los clientes remotos
  tendrán que autenticarse.
- La imagen incluye `ffmpeg` (decodifica el audio del navegador) y `tzdata`
  (ajusta `TZ` en `docker-compose.yml`). No incluye micrófono ni altavoz:
  en modo contenedor la voz va por la interfaz web/satélite.

### Resumen ultra-rápido (PC nueva)

```bash
sudo dnf install -y python3 ffmpeg mpv git && \
git clone <tu-repo> sia && cd sia && \
python3 -m venv .venv && source .venv/bin/activate && \
pip install -r requirements-dev.txt -r requirements-voice.txt && \
cp .env.example .env && nano .env && \
uvicorn app.main:app --port 8000
```

### Resumen ultra-rápido (servidor con Docker)

```bash
git clone <tu-repo> aria && cd aria && \
cp .env.example .env && nano .env && \
docker compose up -d --build
```

## A.R.I.A en tu celular

### Opción 1: App nativa Android (recomendada)

La app **A.R.I.A** (Capacitor) lleva la interfaz de voz dentro y se conecta por
WebSocket a tu PC: sin certificados, sin advertencias del navegador y con
icono propio. Además abre un **puente de control** (`/ws/device`) para que
A.R.I.A opere tu celular (ver "Controla tu celular con SIA"). APK listo en
`~/APKS/A.R.I.A-v2.6.apk`.

**Instalar y usar:**

1. Copia `A.R.I.A-v2.6.apk` al celular e instálala (permite "fuentes desconocidas").
2. En la PC levanta el servidor y abre el firewall:

   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   sudo firewall-cmd --add-port=8000/tcp --permanent && sudo firewall-cmd --reload
   ```

3. Abre A.R.I.A en el celular y listo: **viene pre-conectada de fábrica** a tu
   PC (`192.168.1.25:8000` por WiFi; si no responde, conmuta sola a Tailscale).
   Pide el micrófono en la primera apertura. El engranaje ⚙ permite cambiar la
   dirección después si la IP de tu PC cambia.

**Reconstruir la app** (tras mejorar la interfaz web):

```bash
python scripts/sincronizar_movil.py     # interfaz web → mobile/www
cd mobile
npm install                             # solo la primera vez
npx cap sync android                    # copia www + plugins al proyecto Android
cd android && ./gradlew assembleDebug   # APK en app/build/outputs/apk/debug/
```

> Requiere Node 18+, JDK 17+ y el SDK de Android (`sdk.dir` en
> `mobile/android/local.properties`). Para publicar:
> `./gradlew assembleRelease`.

### Opción 2: Navegador del celular

Sin instalar nada, vía HTTPS con certificado autofirmado:

```bash
python scripts/sia_movil.py    # genera certificado TLS y sirve por https://<IP>:8000
```

Luego abre `https://<IP-DE-TU-PC>:8000`, acepta el aviso de seguridad
("Avanzado → Continuar") y opcionalmente usa "Añadir a pantalla de inicio".
La IP puede cambiar entre redes; regenera con `--regenerar-cert`.

### Usar SIA fuera de casa (Tailscale)

Con una VPN gratuita puedes hablarle a SIA desde cualquier parte del mundo
(datos móviles incluidos), **sin abrir puertos del router**:

1. **Ponle un candado** (imprescindible al salir de tu red). En `.env`:

   ```bash
   ACCESS_TOKEN=$(openssl rand -hex 16)   # o escribe uno a mano y reinicia el servidor
   ```

   El token solo se pide a clientes **remotos** (celular, otro equipo, IP
   Tailscale): desde la PC (`localhost`) SIA nunca lo pide. La app del
   celular lo solicitará la primera vez y lo guardará.

2. **Instala Tailscale** (gratis hasta 100 dispositivos) en tu PC y en el
   celular, con la misma cuenta: <https://tailscale.com/download>

   ```bash
   sudo dnf install tailscale -y
   sudo systemctl enable --now tailscaled
   sudo tailscale up
   tailscale ip -4        # IP fija de tu PC dentro de la VPN: 100.x.y.z
   ```

3. **En la app S.I.A**: toca ⚙ → cambia la dirección a `100.x.y.z:8000` →
   CONECTAR. Listo: funciona igual que en casa, estés donde estés.
   (No necesitas abrir el firewall para Tailscale.)

4. **Que SIA siempre esté lista** — instálala como servicio de tu sesión:

   ```bash
   mkdir -p ~/.config/systemd/user
   cp scripts/sia.service ~/.config/systemd/user/
   systemctl --user daemon-reload
   systemctl --user enable --now sia      # arranca ya y tras cada reinicio
   sudo loginctl enable-linger $USER      # sigue activa aunque cierres sesión
   journalctl --user -u sia -f            # ver sus logs en vivo
   ```

> Nota: la PC debe estar encendida para responder. Tailscale también ofrece
> "SSH" si algún día quieres administrarla remotamente.

### Controla tu celular con SIA

Con la app S.I.A instalada, SIA puede operar tu teléfono: **hacer llamadas**,
**preparar mensajes de WhatsApp o SMS**, **correos**, **abrir aplicaciones**,
**buscar contactos**, **poner alarmas**, **navegar en Maps**, **encender la
linterna**, **vibrarlo**, **copiar al portapapeles**, **ajustar el volumen**,
**decirte la batería** y **mandarte notificaciones**. Ejemplos por voz:

- *"Llama a mamá"* → busca el contacto y marca.
- *"Manda un WhatsApp a Juan diciendo que ya voy"* → abre el chat con el
  texto listo (WhatsApp no permite enviar sin tocar; solo confirmas).
- *"Escribe un correo a ana@ejemplo.com con asunto factura"* → lo deja listo.
- *"Pon una alarma a las 7:30 para trabajar"* → queda en su reloj.
- *"Llévame al aeropuerto"* → abre la navegación en Maps.
- *"Enciende la linterna" / "Vibra mi celular"* → lo encuentra a oscuras.
- *"¿Cuánta batería tiene mi celular?"* → te lo dice.
- *"Copia esta dirección en mi celular"* → va directo a su portapapeles.
- *"Súbeme el volumen del teléfono"*.
- *"Avísame en el celular cuando termine el build"* → notificación local.

Cómo funciona: la app mantiene un segundo WebSocket contra `/ws/device`
(reconexión automática). Las herramientas de teléfono le envían comandos y
el plugin nativo los ejecuta. La primera vez Android pedirá permisos de
teléfono, agenda y notificaciones. Si nadie tiene la app abierta, SIA avisa:
*"No tengo ningún celular conectado."*

> Reconstruir tras cambios nativos: `python scripts/sincronizar_movil.py`,
> luego `npx cap sync android` y `./gradlew assembleDebug` en `mobile/`.

### Satélite de presencia con ESP32 (solo la placa, sin cables)

Un ESP32 **sin ningún hardware extra** se convierte en el radar de A.R.I.A:
escanea Bluetooth buscando tu celular y le avisa al servidor si estás en
casa. Ella lo usa para saludarte cuando llegas ("Bienvenido, jefe") y para
saber tu paradero con su herramienta `get_presence`.

1. **Descubre tu MAC**: en `firmware/satelite_presencia/satelite_presencia.ino`
   pon `MODO_DESCUBRIR = true` + tu WiFi, sube al ESP32 (cable USB-C con
   datos) y lee el Serial Monitor (115200): lista los BLE cercanos con MAC y
   nombre. Copia tu celular a `OBJETIVOS_MAC` o `OBJETIVOS_NOMBRE`, vuelve a
   subir con `MODO_DESCUBRIR = false`.
2. Librerías: WebSockets (Markus Sattler) + ArduinoJson. El escaneo BLE ya
   viene con el soporte ESP32 del Arduino IDE.
3. Listo: LED fijo = estás en casa · parpadeo lento = fuera · latido =
   reconectando. La histéresis (3 escaneos sin verte ≈ 2 min) evita falsas
   salidas por los apagones de Bluetooth del celular.

> Si tu celular rota su dirección BLE aleatoria, rastrea por NOMBRE o usa la
> MAC de un smartwatch (fija).

### Satélite de voz con ESP32 (SIA en cada cuarto)

Un ESP32 con micrófono y bocina se convierte en un "Alexa casero" conectado a
tu SIA: mantienes un botón, hablas, y sueltas — SIA responde **hablando** por
la bocina del cuarto donde estés. Mismo cerebro, misma memoria.

**Materiales** (~$10):

| Pieza | Función |
|---|---|
| ESP32 WROOM | El cerebro pequeño |
| INMP441 | Micrófono I2S |
| MAX98357A | Amplificador I2S para bocina 4-8Ω |
| Botón + cables | Push-to-talk |

**Cableado:**

```
INMP441      VDD→3.3V   GND→GND   L/R→GND    SCK→D14   WS→D15   SD→D32
MAX98357A    VIN→5V     GND→GND   BCLK→D26   LRC→D25   DIN→D22
BOTÓN        D4 ↔ GND   (sin resistencia, INPUT_PULLUP)
```

**Montaje (Arduino IDE):**

1. Instala soporte ESP32 (Gestor de placas → "esp32 by Espressif") y las
   librerías **WebSockets** (Markus Sattler) y **ArduinoJson**.
2. Abre `firmware/satelite_sia/satelite_sia.ino` y edita el bloque de
   configuración: WiFi, IP del PC (o su IP Tailscale) y el ACCESS_TOKEN si
   definiste uno.
3. Placa "ESP32 Dev Module" → Subir. Abre el Serial Monitor (115200) para ver
   la conexión.

El LED integrado te guía: latido = buscando red · fijo = grabando ·
parpadeo triple = error · apagado = lista.

## Uso

### Chat por terminal

```bash
python scripts/chat.py [--reset-memory]
```

### Modo voz (¡como JARVIS!)

```bash
python scripts/voice.py [--no-wake] [--reset-memory]
```

Dile "ARIA" y habla. Responde hablando y ejecutando herramientas.
Dile "salir" para terminar. Sin `--no-wake` responde a todo lo que escuche.

### GUI web

```bash
uvicorn app.main:app --reload
# abre http://localhost:8000
```

Chat en el navegador con confirmaciones interactivas de herramientas.

## Herramientas

| Herramienta           | Permiso | Qué hace                                          |
|-----------------------|---------|---------------------------------------------------|
| `get_time`            | SAFE    | Fecha y hora local                                |
| `get_system_info`     | SAFE    | SO, hostname, CPU, Python                         |
| `get_weather`         | SAFE    | Clima actual de una ciudad                        |
| `search_wikipedia`    | SAFE    | Resumen del primer resultado                      |
| `web_search`          | SAFE    | Búsqueda general en internet (DuckDuckGo)         |
| `remember`            | SAFE    | Guarda un dato a largo plazo                      |
| `media_control`       | SAFE    | Play/pausa/pistas/volumen del sistema             |
| `phone_contacts`      | SAFE    | Busca contactos en la agenda del celular          |
| `open_website`        | CONFIRM | Abre una URL en el navegador                      |
| `open_app`            | CONFIRM | Abre una aplicación instalada (vía .desktop)      |
| `open_folder`         | CONFIRM | Abre una carpeta en VS Code u otro editor/IDE     |
| `play_music`          | CONFIRM | Reproduce una canción (yt-dlp+mpv o navegador)    |
| `create_folder`       | CONFIRM | Crea carpetas                                     |
| `create_file`         | CONFIRM | Crea archivos con contenido                       |
| `read_file`           | SAFE    | Lee archivos de texto o código                    |
| `list_files`          | SAFE    | Lista el contenido de una carpeta                 |
| `run_command`         | RESTRICTED | Ejecuta comandos bash (tests, git, builds...)  |
| `deep_study`          | SAFE    | Estudia un tema a fondo en internet y lo memoriza |
| `phone_call`          | CONFIRM | Llama desde el celular (número o contacto)        |
| `whatsapp_message`    | CONFIRM | Prepara un WhatsApp en el celular (toca enviar)   |
| `send_email`          | CONFIRM | Prepara un correo en el celular (toca enviar)     |
| `open_phone_app`      | CONFIRM | Abre cualquier app instalada en el celular        |
| `send_sms`            | CONFIRM | Prepara un SMS en el celular (toca enviar)        |
| `set_alarm`           | CONFIRM | Alarma o temporizador en el reloj del celular     |
| `navigate`            | CONFIRM | Navegación de Maps hacia un destino               |
| `phone_status`        | SAFE    | Batería y carga del celular                       |
| `phone_torch`         | SAFE    | Enciende/apaga la linterna                        |
| `phone_vibrate`       | SAFE    | Hace vibrar el celular (encuéntralo)              |
| `phone_clipboard`     | SAFE    | Copia texto al portapapeles del celular           |
| `phone_volume`        | SAFE    | Consulta o fija el volumen multimedia             |
| `notify_phone`        | SAFE    | Notificación local en el celular                  |
| `delete_path`         | RESTRICTED | Elimina archivos o carpetas                    |
| `forget`              | CONFIRM | Elimina un recuerdo                               |

Ejemplos con editores de código: *"abre la carpeta sia en vs code"*,
*"abre ~/proyectos/web en vscodium"*. Si no indicas editor, SIA usa el mejor
instalado (`code`, `codium`, `cursor`, `zed`, `subl`, `kate`, `gedit`).

- `SAFE`: se ejecuta sin preguntar. `CONFIRM`: pregunta al usuario antes de
  ejecutar. `RESTRICTED`: exige confirmación siempre.
- `TOOLS_ENABLED` limita el set (vacío = todas). `TOOLS_AUTO_CONFIRM` omite la
  pregunta para herramientas CONFIRM concretas.

## Memoria

SIA guarda su memoria en SQLite (`DATA_DIR/sia.db`): sesiones de conversación,
mensajes y hechos a largo plazo. Al arrancar retoma la conversación anterior y
los hechos se inyectan en el prompt de sistema. `python scripts/chat.py --reset-memory`
borra todo.

## Personalidad y autoaprendizaje

- **Identidad**: SIA sabe quién es y quién la creó (`SIA_CREATOR_NAME`, por
  defecto Samuel). Tiene carácter propio: leal, directa y con chispa, estilo
  JARVIS. Se define en `app/ai/personality.py`; `LLM_SYSTEM_PROMPT` lo
  reemplaza por completo si quieres una identidad distinta.
- **Autoaprendizaje** (`AUTO_LEARN_ENABLED=true`): tras cada intercambio con
  contenido real, un pase ligero del LLM extrae hechos duraderos sobre ti
  (personas, gustos, proyectos, horarios) y los guarda sola en la memoria.
  No pregunta nada: la próxima vez simplemente "ya lo sabe". Los hechos se
  deduplican (tabla `facts`) y puedes borrarlos con la herramienta `forget`.
- **Curiosidad autónoma** (`AUTO_CURIOSITY_ENABLED=true`): cuando la charla
  toca un tema interesante, SIA propone sola una pregunta de investigación,
  la busca en internet (`web_search` con DuckDuckGo, sin API key), sintetiza
  lo esencial y lo guarda como conocimiento permanente. Presupuesto acotado:
  cooldown de 90 s y máximo 8 investigaciones por hora. También puedes pedirle
  directamente "busca en internet..." y usará `web_search`.
- **Estudio profundo** (`deep_study`): pídele "estudia a fondo X" y genera
  sub-preguntas, busca cada una en la web, resume lo esencial y guarda los
  hechos más importantes en su memoria permanente (física, historia, lo que
  sea). Lo aprendido queda disponible para siempre.
- **Te cuenta lo que aprendió**: los descubrimientos autónomos aún no
  compartidos se inyectan al abrir cada sesión; si la saludas o preguntas
  qué hay de nuevo, te cuenta una de sus investigaciones recientes.
- **Charla espontánea** (`PROACTIVE_ENABLED=true`): SIA toma la iniciativa y
  habla sin que le preguntes — te cuenta sus descubrimientos, suelta datos
  curiosos, te saluda según la hora o pregunta cómo va tu día/proyecto.
  Espera entre 8 y 25 min (aleatorio), máximo 3 comentarios por hora,
  silencio nocturno de 23:00 a 08:00 y nunca interrumpe si estás hablándole.

## Compañera de desarrollo

SIA sabe que Samuel desarrolla software y trabaja como su copiloto técnico:

- `read_file`: lee código, configs y logs de tus proyectos.
- `run_command`: ejecuta comandos bash dentro de tu HOME — correr pruebas
  (`pytest -q`), git (`git status`, `git diff`), instalar dependencias,
  compilar... Pide confirmación siempre (RESTRICTED) y respeta un timeout
  (5-240 s). El comando corre en un hilo aparte: SIA nunca se congela.
- Ejemplos: *"lee el main.py del proyecto sia"*, *"corre los tests del
  proyecto sia"*, *"qué archivos cambié en git?"*.

## Velocidad

La voz va primero: la respuesta del LLM se consume en streaming y cada oración
se sintetiza apenas está lista (suenan las primeras palabras mientras el modelo
sigue escribiendo). Además, el primer trozo se recorta por comas: la voz sale
antes de que termine la oración completa. Para respuestas más rápidas:

- Modelos sin fase de razonamiento: `LLM_REASONING_EFFORT=none`
  (ej. `qwen/qwen3.6-27b` en Groq ≈ 0.5 s por respuesta corta).
- Un modelo rápido como principal y uno más capaz de respaldo
  (`LLM_FALLBACK_*`) da lo mejor de ambos mundos.
- Transcripción rápida: `STT_PROVIDER=groq` usa Whisper-turbo (~0.3 s) en vez
  del servicio gratuito de Google (~1-2 s). Si tu LLM de respaldo ya es Groq,
  reutiliza esa key sin configurar nada más.

## Voz natural

- Motor por defecto: voces neuronales de Microsoft Edge (`TTS_PROVIDER=auto`),
  la opción gratuita más humana. `TTS_VOICE` cambia de voz.
- SIA varía sutilmente el ritmo (+/-2%) y el tono de cada frase para no sonar
  monocorde, y limpia emojis/markdown/URLs antes de hablar para no deletrear símbolos.

## Interfaz

`http://localhost:8000/` — **AI Core holográfico** en dorado (`#ffb000`) sobre
negro: núcleo multicapa en canvas (anillos girando a distintas velocidades,
líneas radiales, partículas orbitales y núcleo latiente) con estados animados
(EN ESPERA / ESCUCHANDO / PROCESANDO / RESPONDIENDO / SIN ENLACE), HUD lateral
con lecturas del sistema y arranque tipo terminal. Dos formas de hablarle:

- **Voz libre**: escucha continua con VAD adaptativo; el botón de micrófono
  funciona como interrupción (push-to-talk) aunque esté hablando.
- **Texto**: escribe en el panel inferior ("Pregúntame lo que sea…") y Enter;
  usa el mismo pipeline que la voz (misma respuesta hablada y en pantalla).

Extras: blips de interfaz sintetizados (tras el primer toque, respetando
`prefers-reduced-motion`), respuesta mostrada con fundido y auto-ocultada,
y diseño responsive. La app Android (`scripts/sincronizar_movil.py`) lleva
esta misma interfaz dentro.

## Seguridad

- **Nunca** se almacenan secretos en el código: todo va en `.env` (excluido de git).
- Los logs nunca incluyen valores de campos tipo key/token/password/secret
  (ver `Settings.safe_dict()` y `app/core/logging.py`).
- Toda herramienta pasa por `ToolPolicy` antes de ejecutarse; las de riesgo
  piden confirmación explícita.

## Tests

```bash
source .venv/bin/activate
pytest
```

### Proveedores LLM

SIA usa el proveedor de `.env` (`LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`).
Si configuras `LLM_FALLBACK_*`, actúa como **respaldo automático**: cuando el
principal falla (429, caída, error), SIA conmuta al respaldo en la misma
conversación sin cortes. Ejemplo típico: Gemini (principal) + Groq (respaldo).

## Hoja de ruta (completada)

1. ✅ Base del proyecto
2. ✅ Sistema LLM (`LLMProvider` + `AssistantOrchestrator` + CLI de chat)
3. ✅ Sistema de herramientas con permisos (`BaseTool`, `ToolRegistry`, `ToolPolicy`)
4. ✅ Memoria (`MemoryManager` + SQLite + `remember`/`forget`)
5. ✅ Speech-to-Text (`STTProvider` + `GoogleSTTProvider`)
6. ✅ Text-to-Speech (`TTSProvider` + `EdgeTTSProvider`)
7. ✅ Conversación por voz integrada (`VoiceAssistant` + `scripts/voice.py`)
8. ✅ Interfaz gráfica (GUI web con WebSocket y confirmaciones)
9. ✅ Herramientas avanzadas (`get_weather`, `search_wikipedia`)
10. ✅ Optimización, seguridad y empaquetado (`pyproject.toml`, ruff, lint)