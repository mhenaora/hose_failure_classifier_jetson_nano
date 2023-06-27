"""
tflite_exec.py
This script demonstrates how to do real-time image classification
(inferencing) with TensorflowLite for Jetson Nano using IMX519 Camarray.
"""

import os
import cv2
import json
import timeit
import datetime
import argparse
#import tensorflow as tf
import RPi.GPIO as GPIO
import numpy as np
from utils.camera import add_camera_args, Camera
from utils.display import open_window, show_help_text, set_display
from utils.Focuser import Focuser # Focuser for IMX519
from tflite_runtime.interpreter import Interpreter
# from tensorflow.keras.preprocessing.image import load_img,img_to_array,smart_resize
# from tensorflow.python.saved_model import tag_constants,signature_constants
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.models import load_model
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# from tensorflow.python.framework import convert_to_constants

RESIZED_SHAPE = (224, 224) # Resize images
WINDOW_NAME = 'TFConstruplast-Image-Classifier'
MODEL_TFLITE="modelo/mobilnet_5.tflite"
LABELS="dataset/labels.txt"
inPin=12 # Stop button GPIO pin 12 
outPin=13 # Emergency LED

def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time image classification with TF MobilnetV2 '
            'on Jetson Nano')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument('--multicam', dest='multicam',
                        help='Using 4 in 1 cam array by default [False] '
                             'For 1 in 1 set this flag  ',
                        action='store_true')
    parser.add_argument('--graph', help='Name of the .tflite file, if different than mobilnet_5.tflite',
                        default='modelo/mobilnet_5.tflite')
    parser.add_argument('--labels', help='Name of the label file, if different than labels.txt',
                    default='dataset/labels.txt')
    parser.add_argument('--threshold', help='Minimum confidence threshold for fault in hosses detection',
                    default=0.85)
    args = parser.parse_args()

    return args

def load_labels(path):
    with open(path, 'r') as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}

def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image

def classify_image(interpreter, image, labels, height,width,floating_model, top_k=1):
    # Grab frame from video stream
    #Preprocess
    frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    set_input_tensor(interpreter, input_data)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output = np.squeeze(interpreter.get_tensor(output_details['index']))

  # If the model is quantized (uint8 data), then dequantize the results
    if output_details['dtype'] == np.uint8:
        scale, zero_point = output_details['quantization']
        output = scale * (output - zero_point)

    ordered = np.argpartition(-output, top_k)
    return [(i, output[i]) for i in ordered[:top_k]]  


def loop_and_classify(cam, interpreter, labels, height,width,floating_model):
    """Continuously capture images from camera and do classification."""
    show_help = True
    full_scrn = False
    help_text = '"Esc" to Quit, "H" for Help, "F" to Toggle Fullscreen'
    while True:
        # if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
        #     break
        img = cam.read()
        if img is None:
            break
        timestamp = cv2.getTickCount()
        #classify(img, net, labels, do_cropping)
        results=classify_image(interpreter, img, labels, height, width, floating_model, top_k=1)
        #show_top_preds(img, top_probs, top_labels)
        label_id, prob = results[0]
        FRAME_TIME = (cv2.getTickCount() - timestamp) / cv2.getTickFrequency()

        #print("LabelName:",labels[label_id],"ScoreValue:",prob)
        if label_id == 1 or label_id == 3: # Failure label_id
            print("Time Inference: {0} ms Falla: {1} Confidence: {2} %".format(str(round(FRAME_TIME*1000, 3)),labels[label_id],round(prob*100, 4)))
            GPIO.output(outPin,1)
        else:
            print("Time Inference: {0} ms Class: {1} Confidence: {2} %".format(str(round(FRAME_TIME*1000, 3)),labels[label_id],round(prob*100, 4)))
            GPIO.output(outPin,0)
        # if show_help:
        #     show_help_text(img, help_text)
        # #cv2.imshow(WINDOW_NAME, img)
        key = cv2.waitKey(1) #Listens to the keyboard for presses
        if key == 27 or GPIO.input(inPin) == 0:  # ESC key: quit program
            break
        # elif key == ord('H') or key == ord('h'):  # Toggle help message
        #     show_help = not show_help
        # elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
        #     full_scrn = not full_scrn
        #     set_display(WINDOW_NAME, full_scrn)


def main():
    try: 
        args = parse_args()
        graph = args.graph
        print("Model Inference Implemented:",graph)
        multicam = args.multicam
        #print("multicam:",multicam)
        threshold =args.threshold
        #print("threshold:",threshold)
        labels = load_labels(args.labels)
        interpreter = Interpreter(graph)
        interpreter.allocate_tensors()
        _, height, width, _ = interpreter.get_input_details()[0]['shape']
        # Get model details
        input_details = interpreter.get_input_details()
        #output_details = interpreter.get_output_details()
        height = input_details[0]['shape'][1]
        width = input_details[0]['shape'][2]
        #print("Height:",height,"Width:",width)
        floating_model = (input_details[0]['dtype'] == np.float32)
        
        GPIO.setmode(GPIO.BOARD)# GPIO setup in Jetson
        GPIO.setup(inPin,GPIO.IN)
        GPIO.setup(outPin,GPIO.OUT)

        cam = Camera(args)
        focuser=Focuser(7)# I2C Port connected to Jetson
        focuser.set(Focuser.OPT_FOCUS, 800)# Recommended Values Around 700-900  
        
        i2c = "i2cset -y 7 0x24 0x24 0x00"# Access to camera #1-2-3-4
        os.system(i2c) 
        if not cam.isOpened():
            raise SystemExit('ERROR: failed to open camera!')
        
        # open_window(
        #     WINDOW_NAME, 'TF-Construplast-Image-Classifier',
        #     cam.img_width, cam.img_height)
        loop_and_classify(cam, interpreter,labels, height, width ,floating_model)

        #cam.release()
        #cv2.destroyAllWindows()

    except Exception as e:
        print("ERROR: Could not perform inference")
        print("Exception:", e)

    except KeyboardInterrupt:
        cam.release()
        print("Print Keyboard Interrupt")

    finally:   
        # Clean up
        print("Print Finally")
        cam.release()
        GPIO.output(outPin,0)
        GPIO.cleanup()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
