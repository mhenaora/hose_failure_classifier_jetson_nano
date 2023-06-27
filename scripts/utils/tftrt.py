import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import json
import cv2
import numpy as np


def create_inference_engine(model_file):
    # Cargar el grafo de TensorFlow
    with tf.io.gfile.GFile(model_file, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    # Convertir el grafo de TensorFlow a un grafo de TRT optimizado
    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
        precision_mode=trt.TrtPrecisionMode.FP16, max_workspace_size_bytes=100*1028)
    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=model_file, conversion_params=conversion_params)
    trt_graph = converter.convert()
    print('Done Converting to TF-TRT')
    # Cargar el grafo de TRT optimizado en TensorFlow
    tf.compat.v1.import_graph_def(trt_graph, name='')

    # Obtener los tensores de entrada y salida del modelo
    input_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('input:0')
    output_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('output:0')

    # Crear una sesión de TensorFlow
    session = tf.compat.v1.Session()

    return input_tensor, output_tensor, session


def infer(input_tensor, output_tensor, session, image, class_names):
    # # Cargar la lista de nombres de las clases desde el archivo JSON
    # with open(class_names_file, 'r') as f:
    #     class_names = json.load(f)

    # Preprocesar la imagen
    preprocessed_image = preprocess_image(image)

    # Realizar la inferencia en el modelo optimizado con TRT
    outputs = session.run(output_tensor, feed_dict={
                          input_tensor: preprocessed_image})

    # Obtener la clase predicha y el porcentaje de confianza más alto
    class_id = np.argmax(outputs)
    confidence = outputs[0][class_id]

    # Retornar el nombre de la clase y el porcentaje de confianza
    return class_names[class_id], confidence


def create_inference_engine(model_path, output_model_path, precision='FP16'):
    # Cargar el modelo Keras
    model = keras.models.load_model(model_path)

    # Convertir el modelo a un modelo de TRT
    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode=precision)
    converter = trt.TrtGraphConverterV2(input_saved_model_dir=model_path, conversion_params=conversion_params)
    trt_model = converter.convert()
    print("Conversion Finished using precision: ",precision)
    # Guardar el modelo de TRT en un archivo
    with open(output_model_path, 'wb') as f:
        f.write(trt_model)

    # Cargar el modelo de TRT en una sesión de TensorFlow
    tf_config = tf.compat.v1.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_sess = tf.compat.v1.Session(config=tf_config)
    tf.import_graph_def(trt_model, name='')
    input_tensor = tf_sess.graph.get_tensor_by_name('input_1:0')
    output_tensor = tf_sess.graph.get_tensor_by_name('dense_1/Softmax:0')

    return tf_sess, input_tensor, output_tensor

def infer(model_path, image_path, class_names_path, trt_model_path):
    # Cargar los nombres de clase desde un archivo JSON
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)

    # Cargar el modelo de TRT desde un archivo
    with open(trt_model_path, 'rb') as f:
        trt_model = f.read()

    # Crear la sesión de TensorFlow con el modelo de TRT cargado
    tf_config = tf.compat.v1.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_sess = tf.compat.v1.Session(config=tf_config)
    tf.import_graph_def(trt_model, name='')
    input_tensor = tf_sess.graph.get_tensor_by_name('input_1:0')
    output_tensor = tf_sess.graph.get_tensor_by_name('dense_1/Softmax:0')

    # Preprocesar la imagen
    image = preprocess_image(image_path)

    # Realizar la inferencia en el modelo convertido de TensorRT utilizando la API de TensorFlow
    output = tf_sess.run(output_tensor, feed_dict={input_tensor: image})

    # Obtener el índice de la clase con mayor confianza
    class_id = np.argmax(output)

    # Obtener el nombre de la clase correspondiente al índice
    class_name = class_names[class_id]

    # Obtener el porcentaje de confianza del class_id obtenido
    confidence = output[0][class_id]

    return class_name, confidence

def preprocess_image(image):
    # Preprocesar la imagen para que coincida con el formato de entrada del modelo
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32)
    image /= 255.0
    image = np.expand_dims(image, axis=0)
    return image

def classify_live_video(model_file):
    # Crear el objeto de captura de video
    cap = cv2.VideoCapture(0)

    # Crear el motor y el contexto de TRT optimizado
    input_tensor, output_tensor, session = create_inference_engine(model_file)

    while True:
        # Capturar un frame de video
        ret, frame = cap.read()

        # Realizar la inferencia en el frame
        result = infer(input_tensor, output_tensor, session, frame)

        # Mostrar el resultado en la ventana de video
        cv2.putText(frame, result, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Clasificador de Imágenes", frame)

        # Esperar a que el usuario presione la tecla "q" para salir
        if cv2.waitKey(1) == ord("q"):
            break

    # Liberar el objeto de captura de video y cerrar todas las ventanas
    cap.release()
    cv2.destroyAllWindows()

model_path="/home/jetson-construplast/construplast/scripts/modelo/MobileNetV2_2-1"#"modelo/MobileNetV2_2-1"
image_path="/home/jetson-construplast/construplast/scripts/dataset/test-images/img_4-1_4.jpg" #"dataset/test-images/img_2-1_4.jpg"
class_names_path="/home/jetson-construplast/construplast/scripts/dataset/labels_5.json" #"dataset/labels_5.json"
create_inference_engine(model_path,"/home/jetson-construplast/construplast/scripts/modelo")
class_name,confidence=infer(model_path,image_path,class_names_path)

print("Clase: {} - Confianza: {:.2f}%".format(class_name,confidence*100))

