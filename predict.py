import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
import logging
import estimator
import data_loader

def run(args):


    logging.info("loading model\n")
    model = tf.estimator.Estimator(model_fn=estimator.model_fn, model_dir='./model',
        params=args)
    logging.info("model loaded\n")
    logging.info("loading image\n")
    img = cv2.imread(args.image)
    img = cv2.resize(img, data_loader.img_size)
    img_flat = img.flatten()
    img_flat = np.reshape(img_flat, (1, img_flat.shape[0]))
    logging.info("image loaded\n")

    logging.info("prediction in progress\n")
    #input_fn = tf.estimator.inputs.numpy_input_fn(x=img_flat, shuffle=False)
    input_fn = tf.estimator.inputs.pandas_input_fn(x=pd.DataFrame(img_flat), shuffle=False)
    p = model.predict(input_fn)
    for l, each in enumerate(p):
        for k in data_loader.labels_dict.keys():
            if data_loader.labels_dict[k] == each:
                print(k)   
