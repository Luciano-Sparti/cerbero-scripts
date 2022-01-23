#USO: python procesar_dataset.py --dataset (carpeta origen) --encodings (archivo a generar) --detection-method (hog o cnn)

from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os
import subprocess as li

#Pasaje de Argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
	help="directorio en donde esta las imagenes a procesar")
ap.add_argument("-e", "--encodings", required=True,
	help="archivo a generar")
ap.add_argument("-d", "--detection-method", type=str, default="hog",
	help="Modelo de deteccion de rostos a utilizar: hog o cnn")
args = vars(ap.parse_args())


print("[INFO] Cargando lista de personas...")
imagePaths = list(paths.list_images(args["dataset"]))

#Cargar archivo si existe, o crearlo si no.
try:
    data = pickle.load(open(args["encodings"], "rb"))
except (OSError, IOError) as e:
    knownEncodings = []
    knownNames = []
    data = {"encodings": knownEncodings, "names": knownNames}
    pickle.dump(data, open(args["encodings"], "wb"))

#Por cada imagen, asignar el nombre, y medir vectores de la cara
for (i, imagePath) in enumerate(imagePaths):
    print("[INFO] procesando imagen {}/{}".format(i + 1, len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    print("{}".format(imagePath))
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model=args["detection_method"])
    encodings = face_recognition.face_encodings(rgb, boxes)
    
    #Añade nombre y vectores medidos a los datos
    for encoding in encodings:
        data.get("encodings").extend(encodings)
        data.get("names").append(name)

#Guarda los datos en el archivo
print("[INFO] Guardando parametrías...")
with open(args["encodings"],"wb") as f:
    pickle.dump(data,f)

print("[INFO} Finalizado.")
