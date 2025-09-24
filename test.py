import time
import json
import ssl
from paho.mqtt import client as mqtt

# Configuración AWS IoT Core
ENDPOINT = "a1omfl67425kjv-ats.iot.us-east-2.amazonaws.com"
PORT = 8883
CLIENT_ID = "rsp5"
TOPIC = "alertas"

# Certificados
CA_PATH = "/home/lucianadelarosa/Desktop/proyecto-final/AmazonRootCA1.pem"
CERT_PATH = "/home/lucianadelarosa/Desktop/proyecto-final/certificate.pem.crt"
KEY_PATH = "/home/lucianadelarosa/Desktop/proyecto-final/private.pem.key"

# Función callback al conectar
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("✅ Conectado a AWS IoT Core")
    else:
        print(f"❌ Error de conexión, código: {rc}")

# Crear cliente MQTT
client = mqtt.Client(client_id=CLIENT_ID)
client.tls_set(ca_certs=CA_PATH,
               certfile=CERT_PATH,
               keyfile=KEY_PATH,
               tls_version=ssl.PROTOCOL_TLSv1_2)

client.on_connect = on_connect

# Conectar
client.connect(ENDPOINT, PORT, keepalive=60)
client.loop_start()

# 🔹 Ejemplo de envío de datos (simulados, aquí integrarás tu detección real)
while True:
    # Valores que deberás reemplazar con los que calcule tu código de visión
    perclos = 37.5    # porcentaje (0 a 100)
    blinks = 12       # número de parpadeos detectados
    yawns = 2         # número de bostezos
    estado = "SOMNOLENCIA"  # "FATIGA", "SOMNOLENCIA" o "MICROSUEÑO"

    # Crear payload JSON
    payload = {
        "device_id": CLIENT_ID,
        "estado": estado,
        "perclos": perclos,
        "blinks": blinks,
        "yawns": yawns,
        "ts": int(time.time())  # timestamp UNIX
    }

    # Publicar en IoT Core
    client.publish(TOPIC, json.dumps(payload), qos=1)
    print(f"📤 Enviado: {payload}")

    time.sleep(10)  # cada 10 segundos