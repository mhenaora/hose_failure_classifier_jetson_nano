import cv2
import numpy as np
import tensorflow as tf
from utils.tftrt import create_inference_engine,infer,preprocess_image,classify_live_video
model_path="modelo/MobileNetV2_2-1"
image_path="dataset/test-images/img_2-1_4.jpg"
class_names_path="dataset/labels_5.json"
create_inference_engine(model_path,"modelo/")
class_name,confidence=infer(model_path,image_path,class_names_path)

print("Clase: {} - Confianza: {:.2f}%".format(class_name,confidence*100))