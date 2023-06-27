"""
tflite_gui.py
This script demonstrates how to do real-time image classification
(inferencing) with TensorflowLite for Jetson Nano using IMX519 Camarray.
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
from tflite_runtime.interpreter import Interpreter


WINDOW_NAME = 'TFConstruplast-Image-Classifier'
# #GPIO INPUTS
# switch_nc=1
# switch_no=0
# inPin_start = 11  # start button GPIO pin
# inPin_stop = 12  # stop button GPIO pin
# inPin_reset = 13  # reset button GPIO pin
# inPin_SWAN = 15  # SWAN button GPIO pin
# inPin_GAS = 16  # GAS button GPIO pin
# inPin_PRESION = 22  # BICOLOR button GPIO pin
# inPin_BICOLOR = 18  # PRESION button GPIO pin
# #inPin_CRISTAL = 31  # CRISTAL button GPIO pin
# inPin_POWER_ON = 19  # POWER ON button GPIO pin
# inPin_POWER_OFF = 35  # POWER OFF button GPIO pin
# # GPIO OUTPUTS
# outPin_GOOD = 29  # Output for GOOD detection GPIO pin
# outPin_WARNING = 33  # Output for WARNING detection GPIO pin
# outPin_FAILURE = 24  # Output for FAILURE detection GPIO pin
# outPin_SWAN = 31  # Output for SWAN class GPIO pin
# outPin_GAS = 26  # Output for GAS class GPIO pin
# outPin_PRESION = 23  # Output for PRESION class GPIO pin
# outPin_BICOLOR = 32  # Output for BICOLOR class GPIO pin

#GPIO INPUTS
switch_nc=1
switch_no=0
inPin_start = 11  # start button GPIO pin
inPin_stop = 12  # stop button GPIO pin
inPin_reset = 13  # reset button GPIO pin
inPin_SWAN = 15  # SWAN button GPIO pin
inPin_GAS = 16  # GAS button GPIO pin
inPin_PRESION = 22  # BICOLOR button GPIO pin
inPin_BICOLOR = 18  # PRESION button GPIO pin
#inPin_CRISTAL = 31  # CRISTAL button GPIO pin
inPin_POWER_ON = 19  # POWER ON button GPIO pin
inPin_POWER_OFF = 35  # POWER OFF button GPIO pin
# GPIO OUTPUTS
outPin_GOOD = 29  # Output for GOOD detection GPIO pin
outPin_WARNING = 33  # Output for WARNING detection GPIO pin
outPin_FAILURE = 24  # Output for FAILURE detection GPIO pin
outPin_SWAN = 31  # Output for SWAN class GPIO pin
outPin_GAS = 26  # Output for GAS class GPIO pin
outPin_PRESION = 23  # Output for PRESION class GPIO pin
outPin_BICOLOR = 32  # Output for BICOLOR class GPIO pin

def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time image classification with TF MobilnetV2 '
            'on Jetson Nano')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument('--multicam', dest='multicam',
                        help='Using 2 in 1 (--multicam 2-1) cam array by default'
                        'For 4 in 1 (--multicam 4-1) cam array by default'
                             'For 1 in 1 (--multicam 1-1) cam array by default',
                        default="2-1")
    parser.add_argument('--hose', help='Name of the hose', dest='hose',
                        default='SWAN')
    parser.add_argument('--graph', help='Name of the .tflite file',
                        default='model/2.0/MobileNetV2_DataAugmentation_2.0_2-1_SWAN.tflite')
    parser.add_argument('--labels', help='Name of the label file, if different than labels.txt',
                        default='model/2.0/labels.txt')
    parser.add_argument('--threshold1', help='Minimum confidence threshold class 1 for fault in hosses detection',
                        default=0.85)
    parser.add_argument('--threshold2', help='Minimum confidence threshold class 2 for fault in hosses detection',
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


def classify_image(interpreter, image, labels, height, width, floating_model, top_k=2):
    # Grab frame from video stream
    # Preprocess
    frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        # input_data = (np.float32(input_data) - 127.5) / 127.5 # Normalization [-1-1]
        input_data = np.float32(input_data) / 255.  # Normalization [0,1]

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
    


def loop_and_classify(cam, interpreter, labels, height, width, floating_model, multicam, threshold):
    """Continuously capture images from camera and do classification."""
    show_help = True
    full_scrn = False
    help_text = '"Esc" to Quit, "H" for Help, "F" to Toggle Fullscreen'
    i = 1
    frame_time = 0
    time_switch = 0.001
    top_k=3 # Number of maximum probabilities after each classification
    # prev_label,label_first_class,prob_first_class ,prev_prob = None, None, None, None
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

        results = classify_image(
            interpreter, img, labels, height, width, floating_model, top_k=top_k)

        label_first_class, prob_first_class = results[0]
        label_second_class, prob_second_class = results[1]
        if top_k == 3:
            label_third_class, prob_third_class = results[2] # if top_k =3
        key = cv2.waitKey(1)  # Listens to the keyboard for presses
        if key == 27 or GPIO.input(inPin_reset) == 0 or GPIO.input(inPin_stop) == 0: # ESC key % Reset Button: reset for init state run program
            break
        elif key == ord('H') or key == ord('h'):  # Toggle help message
            show_help = not show_help
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)

        # Change Camera
        i += 1
        if multicam == "1-1":
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

        if multicam == "2-1":
            if i == 1:
                i2c = "i2cset -y 7 0x24 0x24 0x01"  # Access to camera #1-2
                os.system(i2c)
            elif i == 2:
                i2c = "i2cset -y 7 0x24 0x24 0x11"  # Access to camera #3-4
                os.system(i2c)
                i = 0
            time.sleep(time_switch)

            # print("LabelName:",labels[label_first_class],"ScoreValue:",prob_first_class)
        if label_first_class == "1" :  # Failure label_first_class
            # print("Time Inference: {0} ms FAILURE: {1} Confidence: {2} %".format(
            #     str(round(frame_time*1000, 3)), labels[label_first_class], round(prob_first_class*100, 4)))
            #print("Class 1: {}, Class 2: {}".format(results[0],results[1]))
            print("Class 1: {}, Class 2: {}, Class 3: {}".format(results[0],results[1],results[2]))
            GPIO.output(outPin_GOOD, 0)  
            GPIO.output(outPin_WARNING, 0)  
            GPIO.output(outPin_FAILURE, 1)  
        elif label_first_class == "0 "or label_first_class == "3": #Warning or Background label_first_class
            # print("Time Inference: {0} ms WARNING: {1} Confidence: {2} %".format(
            #     str(round(frame_time*1000, 3)), labels[label_first_class], round(prob_first_class*100, 4)))
            #print("Class 1: {}, Class 2: {}".format(results[0],results[1]))
            print("Class 1: {}, Class 2: {}, Class 3: {}".format(results[0],results[1],results[2]))
            GPIO.output(outPin_GOOD, 0)  
            GPIO.output(outPin_WARNING, 1)  
            GPIO.output(outPin_FAILURE, 0) 
        else: #Good label_first_class
            # print("Time Inference: {0} ms GOOD: {1} Confidence: {2} %".format(
            #     str(round(frame_time*1000, 3)), labels[label_first_class], round(prob_first_class*100, 4)))
            #print("Class 1: {}, Class 2: {}".format(results[0],results[1]))
            print("Class 1: {}, Class 2: {}, Class 3: {}".format(results[0],results[1],results[2]))
            GPIO.output(outPin_GOOD, 1)  
            GPIO.output(outPin_WARNING, 0)  
            GPIO.output(outPin_FAILURE, 0) 

        frame_time = (cv2.getTickCount() - timestamp) / cv2.getTickFrequency()


def main():
    try:
        cam=None
        state_initial=0
        state_swan=1
        state_gas=2
        state_presion=3
        state_bicolor=4
        state_cristal=5
        state=0
        #Setting Up GPIO Pins
        GPIO.setmode(GPIO.BOARD)  # GPIO setup pin number in Jetson
        GPIO.setup(inPin_start, GPIO.IN)#,pull_up_down=GPIO.PUD_DOWN)
        GPIO.setup(inPin_stop, GPIO.IN)#,pull_up_down=GPIO.PUD_DOWN)
        GPIO.setup(inPin_reset, GPIO.IN)#,pull_up_down=GPIO.PUD_DOWN)
        GPIO.setup(inPin_SWAN, GPIO.IN)#,pull_up_down=GPIO.PUD_DOWN)
        GPIO.setup(inPin_GAS, GPIO.IN)#,pull_up_down=GPIO.PUD_DOWN)
        GPIO.setup(inPin_PRESION, GPIO.IN)#,pull_up_down=GPIO.PUD_DOWN)
        GPIO.setup(inPin_BICOLOR, GPIO.IN)
        GPIO.setup(inPin_POWER_ON, GPIO.IN)
        GPIO.setup(inPin_POWER_OFF, GPIO.IN)
        GPIO.setup(outPin_GOOD,GPIO.OUT)
        GPIO.setup(outPin_WARNING,GPIO.OUT)
        GPIO.setup(outPin_FAILURE,GPIO.OUT)
        GPIO.setup(outPin_SWAN,GPIO.OUT)
        GPIO.setup(outPin_GAS,GPIO.OUT)
        GPIO.setup(outPin_PRESION,GPIO.OUT)
        GPIO.setup(outPin_BICOLOR,GPIO.OUT)

        args = parse_args()
        multicam = args.multicam
        threshold = args.threshold1
        labels = load_labels(args.labels)
        while GPIO.input(inPin_stop) != 0: # While Stop Button is not press
            # key = cv2.waitKey(0)
            # if key == 27:
            #     print("ESC key press")
            #     break
            if state == state_initial: # Reset Mode
                # Clear outputs
                GPIO.output(outPin_GOOD, 0)  
                GPIO.output(outPin_WARNING, 0)  
                GPIO.output(outPin_FAILURE, 0)  
                GPIO.output(outPin_SWAN, 0)  
                GPIO.output(outPin_GAS, 0)  
                GPIO.output(outPin_PRESION, 0)  
                GPIO.output(outPin_BICOLOR, 0)
                print("Current State: {}".format(state))
                print("Initial State: Press a mode")
                print("--------")
                #time.sleep(60)
                if GPIO.input(inPin_SWAN) == 0 and GPIO.input(inPin_start) == 0:
                    state = state_swan #"Swan Run"
                    print("Current State: {}".format(state))
                elif GPIO.input(inPin_GAS) == 0 and GPIO.input(inPin_start) == 0:
                    state = state_gas #"Gas Run"
                    print("Current State: {}".format(state))
                elif GPIO.input(inPin_PRESION) == 0 and GPIO.input(inPin_start) == 0:
                    state = state_presion #"Presion Run"
                    print("Current State: {}".format(state))
                elif GPIO.input(inPin_BICOLOR) == 0 and GPIO.input(inPin_start) == 0:
                    state = state_bicolor #"Bicolor Run"
                    print("Current State: {}".format(state))
                # elif GPIO.input(inPin_CRISTAL) == 0 and GPIO.input(inPin_start) == 0:
                #     state = state_swan #"Cristal Run"
                #     print("Current State: {}".format(state))
            elif state == state_swan:
                GPIO.output(outPin_SWAN, 1)  
                GPIO.output(outPin_GAS, 0)  
                GPIO.output(outPin_PRESION, 0)  
                GPIO.output(outPin_BICOLOR, 0) 
                hose = "SWAN"
                graph = "model/2.0/MobileNetV2_DataAugmentation_2.0_2-1_SWAN.tflite"

                print("For Hose: {} Labels: {}".format(hose, labels))
                print("Model Inference Implemented:", graph)
                print("Failure threshold: {} Camera Mode: {}".format(threshold, multicam))

                # tflite_interpreter
                interpreter_1 = Interpreter(graph)
                interpreter_1.allocate_tensors()
                _, height, width, _ = interpreter_1.get_input_details()[0]['shape']
                # Get model details
                input_details = interpreter_1.get_input_details()
                # output_details = interpreter_1.get_output_details()
                height = input_details[0]['shape'][1]
                width = input_details[0]['shape'][2]
                # print("Height:",height,"Width:",width)
                floating_model = (input_details[0]['dtype'] == np.float32)
                cam = Camera(args)
                focuser = Focuser(7)  # I2C Port connected to Jetson
                # Recommended Values Around 700-900
                focuser.set(Focuser.OPT_FOCUS, 800)

                if not cam.isOpened():
                    raise SystemExit('ERROR: failed to open camera!')

                loop_and_classify(cam, interpreter_1, labels,
                                height, width, floating_model, multicam, threshold)
                cam.release()
                interpreter_1.reset_all_variables()
                state = state_initial                
            elif state == state_gas:
                GPIO.output(outPin_SWAN, 0)  
                GPIO.output(outPin_GAS, 1)  
                GPIO.output(outPin_PRESION, 0)  
                GPIO.output(outPin_BICOLOR, 0)  
                hose = "GAS"
                graph = "model/2.0/MobileNetV2_DataAugmentation_2.0_2-1_GAS"

                print("For Hose: {} Labels: {}".format(hose, labels))
                print("Model Inference Implemented:", graph)
                print("Failure threshold: {} Camera Mode: {}".format(threshold, multicam))

                # tflite_interpreter
                interpreter_1 = Interpreter(graph)
                interpreter_1.allocate_tensors()
                _, height, width, _ = interpreter_1.get_input_details()[0]['shape']
                # Get model details
                input_details = interpreter_1.get_input_details()
                # output_details = interpreter_1.get_output_details()
                height = input_details[0]['shape'][1]
                width = input_details[0]['shape'][2]
                # print("Height:",height,"Width:",width)
                floating_model = (input_details[0]['dtype'] == np.float32)
                cam = Camera(args)
                focuser = Focuser(7)  # I2C Port connected to Jetson
                # Recommended Values Around 700-900
                focuser.set(Focuser.OPT_FOCUS, 800)

                if not cam.isOpened():
                    raise SystemExit('ERROR: failed to open camera!')

                loop_and_classify(cam, interpreter_1, labels,
                                height, width, floating_model, multicam, threshold)
                cam.release()
                interpreter_1.reset_all_variables()
                state = state_initial
            elif state == state_presion:
                GPIO.output(outPin_SWAN, 0)  
                GPIO.output(outPin_GAS, 0)  
                GPIO.output(outPin_PRESION, 1)  
                GPIO.output(outPin_BICOLOR, 0) 
                hose = "PRESION"
                print("NEEDS TO UPDATE THE MODEL FOR PRESION (IS USING GAS MODEL)")
                graph = "model/2.0/MobileNetV2_DataAugmentation_2.0_2-1_GAS"

                print("For Hose: {} Labels: {}".format(hose, labels))
                print("Model Inference Implemented:", graph)
                print("Failure threshold: {} Camera Mode: {}".format(threshold, multicam))

                # tflite_interpreter
                interpreter_1 = Interpreter(graph)
                interpreter_1.allocate_tensors()
                _, height, width, _ = interpreter_1.get_input_details()[0]['shape']
                # Get model details
                input_details = interpreter_1.get_input_details()
                # output_details = interpreter_1.get_output_details()
                height = input_details[0]['shape'][1]
                width = input_details[0]['shape'][2]
                # print("Height:",height,"Width:",width)
                floating_model = (input_details[0]['dtype'] == np.float32)
                cam = Camera(args)
                focuser = Focuser(7)  # I2C Port connected to Jetson
                # Recommended Values Around 700-900
                focuser.set(Focuser.OPT_FOCUS, 800)

                if not cam.isOpened():
                    raise SystemExit('ERROR: failed to open camera!')

                loop_and_classify(cam, interpreter_1, labels,
                                height, width, floating_model, multicam, threshold)
                cam.release()
                state = state_initial    
            elif state == state_bicolor: 
                GPIO.output(outPin_SWAN, 0)  
                GPIO.output(outPin_GAS, 0)  
                GPIO.output(outPin_PRESION, 0)  
                GPIO.output(outPin_BICOLOR, 1)
                hose = "BICOLOR"
                print("NEEDS TO UPDATE THE MODEL FOR BICOLOR (IS USING GAS MODEL)")
                graph = "model/2.0/MobileNetV2_DataAugmentation_2.0_2-1_GAS"

                print("For Hose: {} Labels: {}".format(hose, labels))
                print("Model Inference Implemented:", graph)
                print("Failure threshold: {} Camera Mode: {}".format(threshold, multicam))

                # tflite_interpreter
                interpreter_1 = Interpreter(graph)
                interpreter_1.allocate_tensors()
                _, height, width, _ = interpreter_1.get_input_details()[0]['shape']
                # Get model details
                input_details = interpreter_1.get_input_details()
                # output_details = interpreter_1.get_output_details()
                height = input_details[0]['shape'][1]
                width = input_details[0]['shape'][2]
                # print("Height:",height,"Width:",width)
                floating_model = (input_details[0]['dtype'] == np.float32)
                cam = Camera(args)
                focuser = Focuser(7)  # I2C Port connected to Jetson
                # Recommended Values Around 700-900
                focuser.set(Focuser.OPT_FOCUS, 800)

                if not cam.isOpened():
                    raise SystemExit('ERROR: failed to open camera!')

                loop_and_classify(cam, interpreter_1, labels,
                                height, width, floating_model, multicam, threshold)
                cam.release()
                state = state_initial
    

    except Exception as e:
        print("ERROR: Could not perform inference")
        print("Exception:", e)
    except UnboundLocalError as e:
        print("ERROR: Variable is not defined")
        print("Exception:", e)
    except KeyboardInterrupt:
        print("KeyBoard Interruption")
        if cam != None:
            cam.release()

    finally:
        # Clean up
        print("Classification Finished")
        GPIO.output(outPin_GOOD, 0)  
        GPIO.output(outPin_WARNING, 0)  
        GPIO.output(outPin_FAILURE, 0)  
        GPIO.output(outPin_SWAN, 0)  
        GPIO.output(outPin_GAS, 0)  
        GPIO.output(outPin_PRESION, 0)  
        GPIO.output(outPin_BICOLOR, 0)
        GPIO.cleanup()
        if cam != None:
            cam.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
