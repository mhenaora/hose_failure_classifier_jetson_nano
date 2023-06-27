"""tf_live_construplast.py

This script demonstrates how to do real-time image classification
(inferencing) with Tensorflow native for Jetson Nano.
"""
import os
import cv2
import json
import timeit
import datetime
import argparse
import tensorflow as tf
import numpy as np
import RPi.GPIO as GPIO
from utils.camera import add_camera_args, Camera
from utils.display import open_window, show_help_text, set_display
from utils.Focuser import Focuser
from tensorflow.keras.preprocessing.image import load_img,img_to_array,smart_resize
from tensorflow.python.saved_model import tag_constants,signature_constants
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.python.framework import convert_to_constants
# from decouple import config

# TF_GPU_ALLOCATOR=config("TF_GPU_ALLOCATOR")
# PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=config("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION")
# TF_CPP_VMODULE=config("TF_CPP_VMODULE")
# convert_graph=config("convert_graph")
# convert_nodes=config("convert_nodes")
# trt_engine=config("trt_engine")
# trt_logger=config("trt_logger")

# PIXEL_MEANS = np.array([[[104., 117., 123.]]], dtype=np.float32)
# #DEPLOY_ENGINE = '' # .engine file 
# ENGINE_SHAPE0 = (3, 224, 224) # input shape images
# ENGINE_SHAPE1 = (1000, 1, 1) # output number of classes

RESIZED_SHAPE = (224, 224) # Resize images
WINDOW_NAME = 'TFConstruplast-Image-Classifier'
MODEL="modelo/mobilnet_5"
MODEL_5="modelo/5_clases_model"
MODEL_FP32='modelo/mobilnet_5_FP32'
MODEL_FP16='modelo/mobilnet_5_FP16'
MODEL_BUILD_FP16="modelo/mobilnet_V2_5_FP16"
LABELS="dataset/labels.txt"
LABELS_5="modelo/labels.txt"
LABELS_FULL="dataset/construplast_labels_9.json"
labels=LABELS

#FPS=0

def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time image classification with TF MobilnetV2 '
            'on Jetson Nano')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument('--crop', dest='crop_center',
                        help='crop center square of image for '
                             'inferencing [False]',
                        action='store_true')
    args = parser.parse_args()
    return args

def decode_predictions(preds, top=3, class_list_path=labels):
  if len(preds.shape) != 2 or preds.shape[1] != 5: # your classes number
    raise ValueError('`decode_predictions` expects '
                     'a batch of predictions '
                     '(i.e. a 2D array of shape (samples, 5)). '
                     'Found array with shape: ' + str(preds.shape))
  index_list = json.load(open(class_list_path))
  results = []
  for pred in preds:
    top_indices = pred.argsort()[-top:][::-1]
    result = [tuple(index_list[str(i)]) + (pred[i],) for i in top_indices]
    result.sort(key=lambda x: x[2], reverse=True)
    results.append(result)
  return results


def show_top_preds(img, top_probs, top_labels):
    """Show top predicted classes and softmax scores."""
    x = 10
    y = 40
    for prob, label in zip(top_probs, top_labels):
        pred = '{:.4f} {:20s}'.format(prob, label)
        #cv2.putText(img, pred, (x+1, y), cv2.FONT_HERSHEY_PLAIN, 1.0,
        #            (32, 32, 32), 4, cv2.LINE_AA)
        cv2.putText(img, pred, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0,
                    (0, 0, 240), 1, cv2.LINE_AA)
        y += 20


def classify(img, net, labels, do_cropping):
    """Classify 1 image (crop)."""
    # Get timestamp for calculating actual framerate
    timestamp = cv2.getTickCount()

    image = img
    var1 = lambda x: smart_resize(x, RESIZED_SHAPE)
    image=var1(image)
    image=np.asarray(image,dtype=np.float32).reshape(1,224,224,3)
    image=(image/127.5)-1
    # image=img_to_array(image)
    # image=np.expand_dims(image,axis=0)
    # image = preprocess_input(image)
    
    # inference the (cropped) image
    #tic = timeit.default_timer()
    out = net.predict(image)
    index=np.argmax(out)
    labels=labels[index]
    confidence_score=out[0][index]
    #toc = timeit.default_timer()
    FRAME_TIME = (cv2.getTickCount() - timestamp) / cv2.getTickFrequency()
    #Print Time inference-class-confidence-datetime
    #print("Time Inference: {0} Class: {1} Confidence: {2} % Date Time: {3}".format(str(round(FRAME_TIME, 3)),labels[2:],str(np.round(confidence_score*100))[:-2],datetime.datetime.now()))
    #Print Time inference-class-confidence
    print("Time Inference: {0} Class: {1} Confidence: {2} %".format(str(round(FRAME_TIME, 3)),labels[2:],str(np.round(confidence_score*100))[:-2]))
    
        #print('{:.3f}s'.format(toc-tic))
        #print("Class:",labels[2:],end="")
        #print("Confidence Score:",str(np.round(confidence_score*100))[:-2],"%")
        #labeling = out(image)
    #preds = out[-1].numpy()
    #print('Predicted: {} Time Inference: {}'.format(decode_predictions(out[-1], top=3)[0],str(round(FRAME_TIME, 3))))
    # output top 3 predicted scores and class labels
    #out_prob = np.squeeze(int(out['dense_1'][0]))
    #top_inds = out_prob.argsort()[::-1][:3]
    
    #FPS = 1 / FRAME_TIME

    return #(out_prob[top_inds], labels[top_inds])


def loop_and_classify(cam, net, labels, do_cropping):
    """Continuously capture images from camera and do classification."""
    show_help = True
    full_scrn = False
    help_text = '"Esc" to Quit, "H" for Help, "F" to Toggle Fullscreen'
    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        img = cam.read()
        if img is None:
            break
        classify(img, net, labels, do_cropping)
        #show_top_preds(img, top_probs, top_labels)
        if show_help:
            show_help_text(img, help_text)
        cv2.imshow(WINDOW_NAME, img)
        key = cv2.waitKey(1) #Listens to the keyboard for presses
        if key == 27:  # ESC key: quit program
            break
        elif key == ord('H') or key == ord('h'):  # Toggle help message
            show_help = not show_help
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)


def main():
    try: 
        args = parse_args()
        print("GPU available: ",len(tf.config.list_physical_devices('GPU')))
        device = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(device[0], True)
        tf.config.experimental.set_virtual_device_configuration(device[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

        net=load_model(MODEL,compile=False)
        labels=open(LABELS,"r").readlines()
        
        cam = Camera(args)
        focuser=Focuser(7)# I2C Port connected to Jetson
        focuser.set(Focuser.OPT_FOCUS, 800)# Recommended Values Around 700-900  
        
        i2c = "i2cset -y 7 0x24 0x24 0x00"# Access to camera #1-2-3-4
        os.system(i2c) 
        if not cam.isOpened():
            raise SystemExit('ERROR: failed to open camera!')
        
        open_window(
            WINDOW_NAME, 'TF-Construplast-Image-Classifier',
            cam.img_width, cam.img_height)
        loop_and_classify(cam, net, labels, args.crop_center)

        #cam.release()
        #cv2.destroyAllWindows()

    except Exception as e:
        print("ERROR: Could not perform inference")
        print("Exception:", e)

    except KeyboardInterrupt:
        cam.release()

    finally:   
        # Clean up
        cam.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
