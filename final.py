#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tempfile import NamedTemporaryFile
import argparse
import cv2

def select_camera():
    found = []
    for i in range(1600):
        cap = cv2.VideoCapture(i)
        success, image = cap.read()
        if success:
            found.append(i)
        cap.release()
    return found

def choose_camera():
    found = select_camera()
    print('Select a camera id:')
    for i in found:
        print('> %d' % i)
    return input('[%d]: ' % found[0])

class Camera(object):
    def __init__(self, cam_id=0):
        self.img_file = None
        self.cam_id = cam_id

    def capture(self):
        if self.img_file:
            os.unlink(self.img_file)
        with NamedTemporaryFile(delete=False, suffix='.jpg') as f:
            self.img_file = f.name
        cam = cv2.VideoCapture(self.cam_id)
        ret, image = cam.read()
        cv2.imwrite(self.img_file, image)
        return self.img_file

    def __del__(self):
        if self.img_file:
            os.unlink(self.img_file)

class PlasticRecyclableClassifier(object):
    recycle = ['milk_carton', 'plastic_bottle', 'pill_bottle']
    non_recycle = ['trash_bags', 'shopping_bags', 'Styrofoam', 'ziplock']

    def __init__(self, data_dir, load_saved_model=False):
        batch_size = 32
        self.img_height = 180
        self.img_width = 180
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
          data_dir,
          validation_split=0.2,
          subset="training",
          seed=123,
          image_size=(self.img_height, self.img_width),
          batch_size=batch_size)
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
          data_dir,
          validation_split=0.2,
          subset="validation",
          seed=123,
          image_size=(self.img_height, self.img_width),
          batch_size=batch_size)

        self.class_names = train_ds.class_names

        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

        normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

        normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
        image_batch, labels_batch = next(iter(normalized_ds))
        first_image = image_batch[0]

        self.model = Sequential([
          layers.experimental.preprocessing.Rescaling(1./255, input_shape=(self.img_height, self.img_width, 3)),
          layers.Conv2D(16, 3, padding='same', activation='relu'),
          layers.MaxPooling2D(),
          layers.Conv2D(32, 3, padding='same', activation='relu'),
          layers.MaxPooling2D(),
          layers.Conv2D(64, 3, padding='same', activation='relu'),
          layers.MaxPooling2D(),
          layers.Flatten(),
          layers.Dense(128, activation='relu'),
          layers.Dense(len(self.class_names))
        ])

        self.model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

        if os.path.exists('./final_weights.index') and load_saved_model:
            self.model.load_weights('final_weights')
        else:
            epochs=10
            history = self.model.fit(
              train_ds,
              validation_data=val_ds,
              epochs=epochs
            )
            self.model.save_weights('final_weights')

    def predict(self, filename):
        img = keras.preprocessing.image.load_img(file_path, target_size=(self.img_height, self.img_width))
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = self.model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        return self.class_names[np.argmax(score)], 100 * np.max(score)

    def recyclable(self, filename):
        class_name, score = self.predict(file_path)
        if class_name in self.non_recycle:
            return (class_name, score, False)
        elif class_name in self.recycle:
            return (class_name, score, True)
        else:
            return (class_name, score, None)

def format_class_name(class_name):
    return ' '.join(class_name.split('_')).title()

def format_recyclable(recyclable):
    if recyclable == True:
        return 'recyclable'
    elif recyclable == False:
        return 'not recyclable'
    elif recyclable == None:
        return 'unknown whether it is recyclable'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-saved-model', action='store_true', help='Load a saved model')
    default_data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'plastics')
    parser.add_argument('--data-dir', help='Location of the plastics data directory', default=default_data_dir)
    parser.add_argument('--from-camera', help='Classify images from the camera instead of filenames', action='store_true')
    parser.add_argument('--camera-id', help='Specify the camera id to read images from')

    args = parser.parse_args()

    classifier = PlasticRecyclableClassifier(args.data_dir, args.load_saved_model)
    if args.from_camera:
        if not args.camera_id:
            args.camera_id = choose_camera()
        cam = Camera(int(args.camera_id))

    while True:
        if args.from_camera:
            with NamedTemporaryFile(delete=False) as f:
                input('Press enter to capture an image from the webcam...')
                file_path = cam.capture()
        else:
            filename = input('Specify an image to predict: ')
            if not filename.strip():
                break
            file_path = os.path.realpath(os.path.expanduser(filename))
            if not os.path.exists(file_path):
                print('File does not exist')
                continue

        class_name, score, recyclable = classifier.recyclable(file_path)
        print('This item appears to be a %s with %.1f confidence, and is %s.' % (format_class_name(class_name), score, format_recyclable(recyclable)))
