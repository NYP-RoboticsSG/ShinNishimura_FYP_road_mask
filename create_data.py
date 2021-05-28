import tensorflow as tf
model = tf.keras.models.load_model('best_masker.tf')

import glob
import cv2
import numpy as np

for filename in glob.glob('data/*jpg'):
    img = cv2.imread(filename)
    img = cv2.resize(img, (128, 128))
    input_img = np.expand_dims(img.astype(np.float32)/255, axis=0)
    result = model(input_img, training=False).numpy()[0]
    result = (np.clip(result, 0.0, 1.0)*255).astype(np.uint8)
    cv2.imshow('', result)
    cv2.waitKey(1)

    cv2.imwrite(filename, result)



