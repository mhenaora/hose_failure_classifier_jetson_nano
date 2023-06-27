"""
Video Recorder for Jetson Nano using IMX519 Camarray.
"""

import os
import cv2
import json
import time
import timeit
import datetime
import argparse
import RPi.GPIO as GPIO
import numpy as np
from utils.camera import add_camera_args, Camera
from utils.display import open_window, show_help_text, set_display
from utils.Focuser import Focuser  # Focuser for IMX519

video_records = 'records/'
inPin = 12  # Stop button GPIO pin 12
outPin = 13  # Emergency LED

import cv2
import os
import datetime

def guardar_video(nombre_archivo, cam):
    # Configuración de la captura de video
    captura = cam
    time_switch =0.001
    # Obtener parámetros de la cámara
    fps = 30 #int(captura.get(cv2.CAP_PROP_FPS))
    resolucion = (640,320)#(int(captura.get(cv2.CAP_PROP_FRAME_WIDTH)), int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    
    # Configuración del archivo de salida
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    archivo_salida = cv2.VideoWriter(video_records+nombre_archivo, codec, fps, resolucion)
    
    # Variables para el control del bucle de grabación
    grabando = True
    tiempo_inicio = datetime.datetime.now()
    
    # Bucle de grabación
    while grabando:
        ret, frame = captura.read()
        if not ret:
            break
        archivo_salida.write(frame)
        
        # Comprobación de si se ha presionado el botón de finalizar grabación
        k = cv2.waitKey(1)
        if k == ord('q'):
            grabando = False
    
    # Cálculo de la duración del video
    tiempo_fin = datetime.datetime.now()
    duracion = tiempo_fin - tiempo_inicio
    
    # Cierre de la captura y el archivo de salida
    captura.release()
    archivo_salida.release()
    
    # Devolución de la duración del video en segundos
    duracion_segundos = duracion.total_seconds()
    return duracion_segundos


def extraer_frames(cam,multicam,nombre_archivo):
    # Configuración de la captura de video
    captura = cam
    
    # Obtener parámetros de la cámara
    fps = int(captura.get(cv2.CAP_PROP_FPS))
    resolucion = (int(captura.get(cv2.CAP_PROP_FRAME_WIDTH)), int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    
    # Crear carpeta para los frames
    fecha_actual = datetime.datetime.now().strftime("%Y-%m-%d")
    ruta_carpeta = f"frames_{fecha_actual}"
    if not os.path.exists(ruta_carpeta):
        os.makedirs(ruta_carpeta)
    
    if i == 1 and multicam == "1-1":
        i2c = "i2cset -y 7 0x24 0x24 0x02"  # Access to camera #1
        os.system(i2c)
    elif i == 1 and multicam == "2-1":
        i2c = "i2cset -y 7 0x24 0x24 0x01"  # Access to camera #1-2
        os.system(i2c)
    elif i == 1 and multicam == "4-1":
        i2c = "i2cset -y 7 0x24 0x24 0x00"  # Access to camera #1-2-3-4
        os.system(i2c)

    # Variables para el control del bucle de extracción de frames
    extrayendo = True
    contador_frames = 0
    GPIO.output(outPin, 1) # Start Recording LED indicator 
    # Bucle de extracción de frames
    while extrayendo:
        ret, frame = captura.read()
        if not ret:
            break
        
        # Guardar frame en archivo
        nombre_frame = f"{ruta_carpeta}/frame_{contador_frames:04d}.jpg"
        cv2.imwrite(nombre_frame, frame)
        contador_frames += 1
        
        # Comprobación de si se ha presionado el botón de finalizar extracción
        k = cv2.waitKey(1)
        if k == ord('q')or GPIO.input(inPin) == 0:
            extrayendo = False
            GPIO.output(outPin,0)

        # Change Camera
        i += 1
        if multicam =="1-1":
            if i == 1:
                i2c = "i2cset -y 7 0x24 0x24 0x02"  # Access to camera #1
                os.system(i2c)
            elif i == 2:
                i2c = "i2cset -y 7 0x24 0x24 0x12"  # Access to camera #2
                os.system(i2c)
            elif i == 3:
                i2c = "i2cset -y 7 0x24 0x24 0x22"  # Access to camera #3
                os.system(i2c)
            elif i == 4:
                i2c = "i2cset -y 7 0x24 0x24 0x32"  # Access to camera #4
                os.system(i2c)
                i = 0
            time.sleep(time_switch)

        if multicam =="2-1":
            if i == 1:
                i2c = "i2cset -y 7 0x24 0x24 0x01"  # Access to camera #1-2
                os.system(i2c)
            elif i == 2:
                i2c = "i2cset -y 7 0x24 0x24 0x11"  # Access to camera #3-4
                os.system(i2c)
                i = 0
            time.sleep(time_switch)

    # Cierre de la captura
    captura.release()
    
    # Devolución del número de frames extraídos
    return contador_frames


def parse_args():
    """Parse input arguments."""
    desc = ('Video Recorder for Jetson Nano using IMX519 Camarray')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument('--multicam', dest='multicam',
                        help='Using 2 in 1 (--multicam 2-1) cam array by default'
                        'For 4 in 1 (--multicam 4-1) cam array by default'
                             'For 1 in 1 (--multicam 1-1) cam array by default',
                        default="2-1")
    parser.add_argument('--file', help='Video File Name',default="video_")
    parser.add_argument('--focus', help='Focus Value for Video Recording (From 0-1000)',
                        default=800)
    args = parser.parse_args()

    return args


def loop_record(cam,multicam):
    """Continuously capture images from camera and do classification."""
    show_help = True
    full_scrn = False
    help_text = '"Esc" to Quit, "H" for Help, "F" to Toggle Fullscreen'
    i = 1
    frame_time = 0
    time_switch =0.005
   
    if i == 1 and multicam == "1-1":
        i2c = "i2cset -y 7 0x24 0x24 0x02"  # Access to camera #1
        os.system(i2c)
    elif i == 1 and multicam == "2-1":
        i2c = "i2cset -y 7 0x24 0x24 0x01"  # Access to camera #1-2
        os.system(i2c)
    elif i == 1 and multicam == "4-1":
        i2c = "i2cset -y 7 0x24 0x24 0x00"  # Access to camera #1-2-3-4
        os.system(i2c)

    while True:
        timestamp = cv2.getTickCount()
        # if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
        #     break
        img = cam.read()
        if img is None:
            break
        
        # if show_help:
        #     show_help_text(img, help_text)
        # cv2.imshow(WINDOW_NAME, img)
        key = cv2.waitKey(1)  # Listens to the keyboard for presses
        if key == 27 or GPIO.input(inPin) == 0:  # ESC key: quit program
            break
        elif key == ord('H') or key == ord('h'):  # Toggle help message
            show_help = not show_help
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)

        # Change Camera
        i += 1
        if multicam =="1-1":
            if i == 1:
                i2c = "i2cset -y 7 0x24 0x24 0x02"  # Access to camera #1
                os.system(i2c)
            elif i == 2:
                i2c = "i2cset -y 7 0x24 0x24 0x12"  # Access to camera #2
                os.system(i2c)
            elif i == 3:
                i2c = "i2cset -y 7 0x24 0x24 0x22"  # Access to camera #3
                os.system(i2c)
            elif i == 4:
                i2c = "i2cset -y 7 0x24 0x24 0x32"  # Access to camera #4
                os.system(i2c)
                i = 0
            time.sleep(time_switch)

        if multicam =="2-1":
            if i == 1:
                i2c = "i2cset -y 7 0x24 0x24 0x01"  # Access to camera #1-2
                os.system(i2c)
            elif i == 2:
                i2c = "i2cset -y 7 0x24 0x24 0x11"  # Access to camera #3-4
                os.system(i2c)
                i = 0
            time.sleep(time_switch)

        frame_time = (cv2.getTickCount() - timestamp) / cv2.getTickFrequency()

def main():
    try:
        args = parse_args()

        #graph = args.graph
        nombre_archivo=args.file
        nombre_archivo=nombre_archivo+datetime.datetime.now().strftime("%Y-%m-%d")
        multicam = args.multicam
        focus = args.focus
        # labels = load_labels(args.labels)
        # print("Model Inference Implemented:", graph)
        print("Focus Value: {} Camera Mode: {} File Name: {}".format(focus, multicam,nombre_archivo))

        GPIO.setmode(GPIO.BOARD)  # GPIO setup in Jetson
        GPIO.setup(inPin, GPIO.IN)
        GPIO.setup(outPin, GPIO.OUT)

        cam = Camera(args)
        focuser = Focuser(7)  # I2C Port connected to Jetson
        # Recommended Values Around 700-900
        focuser.set(Focuser.OPT_FOCUS, focus)

        if not cam.isOpened():
            raise SystemExit('ERROR: failed to open camera!')

        # open_window(
        #     WINDOW_NAME, 'TF-Construplast-Image-Classifier',
        #     cam.img_width, cam.img_height)
        #loop_record(cam, multicam)
        extraer_frames(cam,multicam,nombre_archivo)
        # cam.release()
        # cv2.destroyAllWindows()

    except Exception as e:
        print("ERROR: Could not perform inference")
        print("Exception:", e)

    except KeyboardInterrupt:
        print("KeyBoard Interruption")
        cam.release()

    finally:
        # Clean up
        print("Print Finally")
        GPIO.output(outPin, 0)
        GPIO.cleanup()
        cam.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
