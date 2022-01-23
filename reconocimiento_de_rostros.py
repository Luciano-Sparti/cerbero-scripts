##USO: python3 reconocimiento_de_rostros.py --cascada modelo.xml --encodings dataset.pkl

##Importa librerias necesarias
import face_recognition
import argparse
import imutils
import pickle
import cv2
#import mysql.connector
import numpy as np
import threading
from flask import Flask, render_template, Response

##Control de argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascada", required=True, help = "ruta del archivo detector de cascadas HAAR")
ap.add_argument("-e", "--encodings", required=True, help="ruta a los encodings serializados")

args = vars(ap.parse_args())

##Funcion para hacer lindo el texto por consola
def esc(code):
    return f'\033[{code}m'

##Manejo de threads para optimizar sistema
outputFrame = None
lock = threading.Lock()

##Incializar Flask para ver en web
print( esc('93') + "[INFO] " + esc('0') + "Iniciando servicios web...")
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

def generate():
    global outputFrame, lock
    while True:
        with lock:
            if outputFrame is None:
                continue

            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            if not flag:
                continue

        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@app.route("/accederCamara")
def video_feed():
    return Response(generate(), mimetype = "multipart/x-mixed-replace; boundary=frame")


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=1024, debug=True, threaded=True, use_reloader=False)

print( esc('93') + "[INFO] " + esc('0') + "Cargando datos de reconocimiento facial...")
##Determinar que método cascada utilizar según parámetro
detector = cv2.CascadeClassifier(args["cascada"])

##Carga encodings
data = pickle.loads(open(args["encodings"], "rb").read())

print( esc('93') + "[INFO] " + esc('0') + "Iniciando stream de cámara...")
##Carga de video (RTSP) en Thread
#video_getter = VideoGet("rtsp://cerbero:cerbero@200.61.187.65:7771/11").start()
##Carga de Video (webcam - USB)
captura = cv2.VideoCapture(0)

if not captura.isOpened():
    print("Error. No se puede acceder a la camara")
    exit()

print(esc('93') + "[INFO] " + esc('94;1;3') + "CERBERO" + esc(0) + " iniciado.")

while True:
    ret, frame = captura.read()

    ##Cambiamos el tamaño para que procese más rápido
    frame = imutils.resize(frame, width=600)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    timestamp = datetime.datetime.now()
    cv2.putText(frame, timestamp.strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    ##Detecto rostros en el frame a blanco y negro para detectar mejor el contraste
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    ##Utiliza biometrías para detección de rostros con imágen RGB.
    boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []

    for encoding in encodings:
        name = "Desconocido"
        matches = face_recognition.compare_faces(data["encodings"],encoding)

        ##Si encuentra similitud entre las mediciones en vivo y los encodings
        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            ##Por cada similitud, añade votos y se los asigna al nombre asignado al encode.
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
                lcount = counts.get(name, 0)
        with lock:
            outputFrame = frame.copy()

        ##Determinar persona por votacion, según el conteo anterior.
        name = max(counts, key=counts.get)

        ##Le asigna un nombre a la detección votada.
        names.append(name)

        ##Dibuja sobre el frame
        for (x, y, w, h) in rects:
            cv2.rectangle(frame, (x, y), (x+w, y+h),(0, 255, 0), 2)
            cv2.putText(frame, name, (h, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        ##Añade registros y activa procesos de reconocimiento exitoso
        ##Requiere 8 reconocimientos ciertos y consecutivos para determinar la veracidad del reconocimeinto.
        if name != "Desconocido" and counts.get(name,0) > 8:
            print (esc('32') + "[SUCC] " + esc('0') + "Biométrica reconocida: " + esc('34;1;3') + name + esc('0'))

    ##Muestra el frame en una ventana
    cv2.imshow("CERBERO", frame)

    ##Si presiona la letra Q, finaliza la sesión de CERBERO
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        print("\n" + esc('93') + "[INFO] " + esc('0') + "Limpiando antes de salir...")
        cv2.destroyAllWindows()
        print("\n" + esc('32') + "[SUCC] " + esc('0') + "Salida exitosa!")
        break
