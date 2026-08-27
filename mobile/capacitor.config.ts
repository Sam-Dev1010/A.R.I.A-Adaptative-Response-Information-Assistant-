import type { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'com.samuel.sia',
  appName: 'A.R.I.A',
  webDir: 'www',
  server: {
    // http (no https): así el WebView puede abrir ws:// hacia la PC de la red
    // local sin certificados. localhost sigue siendo contexto seguro para el
    // micrófono, así que getUserMedia funciona igual.
    androidScheme: 'http',
    cleartext: true
  }
};

export default config;
