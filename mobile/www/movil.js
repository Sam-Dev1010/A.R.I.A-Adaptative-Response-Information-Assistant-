/* S.I.A móvil: conexión al servidor de tu PC.
   La interfaz es la misma que la versión web; este archivo solo resuelve
   A QUÉ servidor conectarse (guardado en el celular), pide el token de
   acceso remoto y mantiene el PUENTE DE DISPOSITIVO por el que SIA controla
   el celular: llamadas, WhatsApp, correos y abrir aplicaciones. */
(function () {
  "use strict";
  const CLAVE = "sia.servidor";
  const CLAVE_IDX = "sia.servidor_idx";
  const CLAVE_TOKEN = "sia.token";
  /* Token pre-cargado de fábrica: sincronizar_movil.py lo reemplaza por el
     ACCESS_TOKEN del .env del servidor al generar la app. */
  const DEFECTO_TOKEN = "afe27c8940d6ee04311ff0b2a525e336";

  /* Si el celular aún no tiene token guardado, usa el de fábrica para que la
     app conecte sin escribir nada (el ajuste manual siempre puede cambiarlo). */
  function prepararToken() {
    if (DEFECTO_TOKEN.includes("__")) return;
    if (!localStorage.getItem(CLAVE_TOKEN)) {
      localStorage.setItem(CLAVE_TOKEN, DEFECTO_TOKEN);
    }
  }

  /* Servidores por defecto: la app viene PRE-CONECTADA de fábrica.
     Prueba primero la IP local de la PC y, si no responde (fuera de casa),
     conmuta sola a la IP de Tailscale. El ajuste manual (⚙) siempre gana. */
  const DEFECTO = [
    "http://192.168.1.25:8000",      // PC en la red WiFi de casa
    "http://100.124.157.109:8000",   // PC por Tailscale (desde cualquier parte)
  ];

  function leerIdx() {
    return parseInt(localStorage.getItem(CLAVE_IDX) || "0", 10) || 0;
  }

  function actual() {
    return localStorage.getItem(CLAVE) || DEFECTO[leerIdx() % DEFECTO.length];
  }

  /* Sin servidor manual configurado: pasa al siguiente candidato. */
  function rotar() {
    if (localStorage.getItem(CLAVE)) return;
    localStorage.setItem(CLAVE_IDX, String((leerIdx() + 1) % DEFECTO.length));
  }

  function websocket() {
    const base = actual().replace(/\/+$/, "");
    if (!base) return "ws://127.0.0.1:9/ws/interface"; /* aún sin configurar */
    return base.replace(/^http/i, "ws") + "/ws/interface";
  }

  function websocketDispositivo() {
    const base = actual().replace(/\/+$/, "");
    if (!base) return "ws://127.0.0.1:9/ws/device";
    return base.replace(/^http/i, "ws") + "/ws/device";
  }

  function normalizar(texto) {
    let direccion = String(texto || "").trim();
    if (!direccion) return "";
    if (!/^https?:\/\//i.test(direccion)) direccion = "http://" + direccion;
    return direccion.replace(/\/+$/, "");
  }

  function guardar(entrada) {
    const direccion = normalizar(entrada);
    if (!direccion) return false;
    localStorage.setItem(CLAVE, direccion);
    return true;
  }

  window.Servidor = { actual, websocket, websocketDispositivo, guardar, rotar };

  /* Permisos al arrancar: pide el micrófono de una vez (Android muestra el
     diálogo en la primera apertura) para que la voz funcione sin toques extra. */
  function pedirMicrofono() {
    try {
      navigator.mediaDevices
        .getUserMedia({ audio: true })
        .then((stream) => stream.getTracks().forEach((t) => t.stop()))
        .catch(() => {});
    } catch (err) { /* navegador sin getUserMedia: nada que pedir */ }
  }

  /* ================= puente de dispositivo =================
     El servidor puede ordenar acciones al celular por /ws/device.
     Se ejecutan con el plugin nativo SiaBridge (llamadas, WhatsApp,
     correo, abrir apps, agenda, batería, linterna, vibración,
     portapapeles, SMS, alarmas, navegación, volumen y notificaciones);
     en navegador hay respaldos simples. */
  let wsDev = null, reintento = 0;
  const oyentesEstado = [];

  function plugin() {
    return (window.Capacitor && window.Capacitor.Plugins && window.Capacitor.Plugins.SiaBridge) || null;
  }

  async function ejecutar(accion, params) {
    const nativo = plugin();
    if (nativo) {
      if (accion === "call") {
        const r = await nativo.call({ number: params.number || "" });
        return { ok: true, message: r.message || "Llamada iniciada." };
      }
      if (accion === "whatsapp") {
        const r = await nativo.whatsapp({ number: params.number || "", text: params.text || "" });
        return { ok: !!r.ok, message: r.message || "WhatsApp abierto." };
      }
      if (accion === "email") {
        const r = await nativo.email({
          to: params.to || "", subject: params.subject || "", body: params.body || "",
        });
        return { ok: !!r.ok, message: r.message || "Correo listo para enviar." };
      }
      if (accion === "open_app") {
        const r = await nativo.openApp({ name: params.name || "" });
        return { ok: !!r.ok, message: r.message || "" };
      }
      if (accion === "contacts") {
        const r = await nativo.contacts({ query: params.query || "" });
        return { ok: true, message: "", data: { contacts: r.contacts || [] } };
      }
      if (accion === "status") {
        const r = await nativo.deviceStatus({});
        return { ok: !!r.ok, message: r.message || "", data: r.data || {} };
      }
      if (accion === "torch") {
        const r = await nativo.torch({ mode: params.mode || "on" });
        return { ok: !!r.ok, message: r.message || "" };
      }
      if (accion === "vibrate") {
        const r = await nativo.vibrate({ ms: params.ms || 400 });
        return { ok: !!r.ok, message: r.message || "" };
      }
      if (accion === "clipboard") {
        const r = await nativo.clipboardSet({ text: params.text || "" });
        return { ok: !!r.ok, message: r.message || "" };
      }
      if (accion === "sms") {
        const r = await nativo.sms({ number: params.number || "", text: params.text || "" });
        return { ok: !!r.ok, message: r.message || "" };
      }
      if (accion === "alarm") {
        const r = await nativo.setAlarm({
          hour: params.hour, minute: params.minute,
          label: params.label || "SIA", timer: !!params.timer,
        });
        return { ok: !!r.ok, message: r.message || "" };
      }
      if (accion === "navigate") {
        const r = await nativo.navigate({ destination: params.destination || "" });
        return { ok: !!r.ok, message: r.message || "" };
      }
      if (accion === "volume") {
        const r = await nativo.mediaVolume({ level: params.level });
        return { ok: !!r.ok, message: r.message || "" };
      }
      if (accion === "notify") {
        const r = await nativo.notifyMe({
          title: params.title || "SIA", message: params.message || "",
        });
        return { ok: !!r.ok, message: r.message || "" };
      }
      return { ok: false, message: "Acción desconocida: " + accion };
    }

    /* Respaldos web (sin plugin nativo): solo abrir intents básicos. */
    try {
      if (accion === "call" && params.number) {
        location.href = "tel:" + encodeURIComponent(params.number);
        return { ok: true, message: "Abriendo el teléfono…" };
      }
      if (accion === "email" && params.to) {
        const q = new URLSearchParams({ subject: params.subject || "", body: params.body || "" });
        location.href = "mailto:" + encodeURIComponent(params.to) + "?" + q.toString();
        return { ok: true, message: "Correo listo para enviar." };
      }
    } catch (err) { /* sigue como error */ }
    return { ok: false, message: "Esta acción necesita la app S.I.A instalada." };
  }

  function notificarEstado(conectado) {
    for (const fn of oyentesEstado) fn(conectado);
  }

  function conectarDispositivo() {
    if (!actual()) return;
    clearTimeout(reintento);
    try { wsDev && wsDev.close(); } catch (err) { /* ya estaba cerrado */ }
    const token = encodeURIComponent(localStorage.getItem("sia.token") || "");
    wsDev = new WebSocket(websocketDispositivo() + (token ? "?token=" + token : ""));
    wsDev.onopen = () => {
      reintento = 0;
      notificarEstado(true);
      wsDev.send(JSON.stringify({
        type: "hello",
        model: (navigator.userAgent.match(/Android[^;)]*/i) || ["Android"])[0].trim(),
      }));
    };
    wsDev.onclose = () => {
      notificarEstado(false);
      reintento = setTimeout(conectarDispositivo, 4000); /* reconexión automática */
    };
    wsDev.onerror = () => { try { wsDev.close(); } catch (err) { /* noop */ } };
    wsDev.onmessage = async (ev) => {
      let msg;
      try { msg = JSON.parse(ev.data); } catch (err) { return; }
      if (msg.type !== "cmd") return;
      let resultado;
      try {
        resultado = await ejecutar(msg.action, msg.params || {});
      } catch (err) {
        resultado = { ok: false, message: String(err && err.message || err) };
      }
      if (wsDev && wsDev.readyState === 1) {
        wsDev.send(JSON.stringify({
          type: "result", id: msg.id,
          ok: !!resultado.ok,
          message: resultado.message || "",
          data: resultado.data || {},
        }));
      }
    };
    setInterval(() => {
      if (wsDev && wsDev.readyState === 1) wsDev.send(JSON.stringify({ type: "ping" }));
    }, 25000);
  }

  window.SiaPuente = {
    onEstado: (fn) => oyentesEstado.push(fn),
    conectado: () => !!(wsDev && wsDev.readyState === 1),
  };

  prepararToken();
  conectarDispositivo();
  pedirMicrofono();

  /* ---------- pantalla de ajustes ---------- */
  const ESTILOS = `
  #sia-config { position: fixed; inset: 0; z-index: 100; background: #010204;
    display: flex; align-items: center; justify-content: center;
    font-family: "Segoe UI", system-ui, sans-serif; color: #ffe3b3; }
  #sia-config .caja { width: min(430px, 88vw); border: 1px solid rgba(255,176,0,.55);
    background: #0a0806; padding: 30px 28px; box-shadow: 0 0 60px rgba(255,176,0,.15); }
  #sia-config h1 { font-size: 22px; letter-spacing: 6px; margin: 0 0 4px; color: #ffd166; }
  #sia-config p.sub { font-size: 11px; letter-spacing: 2px; text-transform: uppercase;
    color: rgba(255,176,0,.55); margin: 0 0 24px; }
  #sia-config label { display: block; font-size: 11px; letter-spacing: 2px;
    text-transform: uppercase; color: rgba(255,176,0,.55); margin-bottom: 8px; }
  #sia-config input { width: 100%; font-size: 16px; padding: 12px 14px;
    background: #010204; border: 1px solid rgba(255,176,0,.55); color: #ffd9a0;
    outline: none; box-sizing: border-box; }
  #sia-config input:focus { border-color: #ffb000; box-shadow: 0 0 14px rgba(255,176,0,.35); }
  #sia-config button { width: 100%; margin-top: 18px; font-size: 14px; letter-spacing: 3px;
    padding: 13px 0; background: rgba(255,176,0,.1); border: 1px solid #ffb000;
    color: #ffb000; text-transform: uppercase; cursor: pointer; font-weight: bold; }
  #sia-config button:active { background: rgba(255,176,0,.3); }
  #sia-config .pista { font-size: 11px; color: rgba(255,227,179,.45); margin-top: 16px;
    line-height: 1.6; }
  #sia-config .estado-puente { margin-top: 12px; font-size: 11px; letter-spacing: 2px;
    text-transform: uppercase; color: rgba(255,227,179,.55); }
  #sia-config .estado-puente b { color: #ffd166; }
  #sia-btn { position: fixed; top: 16px; right: 72px; z-index: 90; width: 34px; height: 34px;
    border-radius: 50%; border: 1px solid rgba(255,176,0,.45); background: rgba(1,2,4,.7);
    color: #ffb000; font-size: 17px; line-height: 1; cursor: pointer; }`;

  function montarEstilos() {
    if (document.getElementById("sia-movil-css")) return;
    const estilo = document.createElement("style");
    estilo.id = "sia-movil-css";
    estilo.textContent = ESTILOS;
    document.head.appendChild(estilo);
  }

  function abrirPanel() {
    montarEstilos();
    document.getElementById("sia-config")?.remove();
    const panel = document.createElement("div");
    panel.id = "sia-config";
    panel.innerHTML = `
      <div class="caja">
        <h1>S.I.A</h1>
        <p class="sub">Enlace con tu computadora</p>
        <label for="sia-direccion">Dirección del servidor</label>
        <input id="sia-direccion" type="text" autocomplete="off"
               autocapitalize="off" spellcheck="false"
               placeholder="192.168.1.25:8000" value="${actual()}">
        <button id="sia-guardar" type="button">Conectar</button>
        <div class="estado-puente" id="sia-estado-puente"></div>
        <div class="pista">
          La app ya viene conectada a tu PC (WiFi de casa o Tailscale).
          Cámbiala aquí si la IP cambia.<br>
          Servidor en la PC: <b>uvicorn app.main:app --host 0.0.0.0 --port 8000</b><br>
          Escribe solo la IP y el puerto. Fuera de casa el servidor te pedirá
          el token (<b>.env → ACCESS_TOKEN</b>).
        </div>`;
    document.body.appendChild(panel);
    const actualizar = () => {
      const destino = document.getElementById("sia-estado-puente");
      if (!destino) return;
      destino.innerHTML = window.SiaPuente.conectado()
        ? "PUENTE DE CONTROL: <b>ENLAZADO</b> · SIA puede operar este celular"
        : "PUENTE DE CONTROL: sin enlace con el servidor";
    };
    window.SiaPuente.onEstado(actualizar);
    actualizar();
    const conectar = () => {
      if (guardar(document.getElementById("sia-direccion").value)) {
        location.reload();
      }
    };
    panel.querySelector("#sia-guardar").addEventListener("click", conectar);
    panel
      .querySelector("#sia-direccion")
      .addEventListener("keydown", (ev) => ev.key === "Enter" && conectar());
  }

  function montarBotonAjustes() {
    montarEstilos();
    if (document.getElementById("sia-btn")) return;
    const boton = document.createElement("button");
    boton.id = "sia-btn";
    boton.type = "button";
    boton.textContent = "⚙";
    boton.addEventListener("click", abrirPanel);
    document.body.appendChild(boton);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () =>
      actual() ? montarBotonAjustes() : abrirPanel()
    );
  } else {
    actual() ? montarBotonAjustes() : abrirPanel();
  }
})();
