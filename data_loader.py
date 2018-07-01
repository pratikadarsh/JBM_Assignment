import numpy as np
import cv2
import pandas as pd
import os
import logging

labels_dict = {"61326_0k_back"     :0,
               "61326_ok_front"    :1,
               "61326_Scratch_Mark":2,
               "61326_Slot_Damage" :3,
               "61326_Thinning"    :4,
               "61326_Wrinkle"  :5}

img_size = (70,70)
image_data = np.empty((1,14701), int)
#image_data = np.empty((1,14701), int)

def get_label(root_dir):
    ''' Returns the label corresponding to a class of image.'''

    label_str = root_dir.split(os.sep)[-1]
    if label_str in labels_dict:
        return labels_dict[label_str]
    else:
        print("key error")

def load_image(root_dir, image_files, label):
    ''' Loads each image into the img array.'''

    global image_data
    logging.info("Parsing through directories for files\n")
    for file_name in image_files:
        if file_name[-4:] == ".jpg":
            file_path = root_dir + os.sep + file_name
            logging.debug("Processing file %s\n", file_path)
            img = cv2.imread(file_path)
            img = cv2.resize(img, img_size)
            img_flat = img.flatten()
            img_flat = np.reshape(img_flat, (1, img_flat.shape[0]))
            img_labld = np.zeros((1, img_flat.shape[1]+1))
            img_labld[:, -1] = label
            img_labld[:, :-1] = img_flat
            image_data = np.append(image_data, img_labld, axis=0)        

def split_into_XY(data):
    ''' Splits a data into features and labels. '''

    return data.iloc[:, :data.shape[1]-1], data.iloc[:, -1]

def split_data(raw_data, split_param):
    ''' Splits a dataframe into Train, Test and Validation sets. '''

    assert sum(split_param) == 1
    logging.info("splitting dataset \n")

    cum_ratio = split_param[0]
    train_size = int(cum_ratio * raw_data.shape[0])
    train = raw_data.iloc[0:train_size, :]
    train_X, train_Y = split_into_XY(train)

    cum_ratio += split_param[1]
    test_size = int(cum_ratio * raw_data.shape[0])
    test = raw_data.iloc[train_size:test_size, :]
    test_X, test_Y = split_into_XY(test)

    cum_ratio += split_param[1]
    validation_size = int(cum_ratio * raw_data.shape[0])
    validation = raw_data.iloc[test_size:validation_size, :]
    validation_X, validation_Y = split_into_XY(validation)

    logging.debug("Dataset shape: %s, Train shape : %s, Test shape: %s,\
        Validation shape : %s\n", raw_data.shape, train.shape, test.shape,
        validation.shape)
    return { "train_X" : train_X,
             "train_Y" : train_Y,
             "test_X" : test_X,
             "test_Y" : test_Y,
             "validation_X" : validation_X,
             "validation_Y" : validation_Y} 

def load_data(split_param):
    ''' Loads images and returns train, test and validation sets. '''

    if os.path.isfile("foo.pkl") == False:
        logging.info("pre-loaded data not present\n")
        logging.info("Parsing through directories for files\n")
        for root, dir, files in os.walk("/home/pratik/JBM_Assignment/All_61326/train_61326"):
            if len(files) != 0:
                logging.info("Processing directory : %s", root)
                label = get_label(root)
                load_image(root, files, label)
            df = pd.DataFrame(image_data[1:, :])
            df.to_pickle("foo.pkl")
    raw_data = pd.read_pickle("foo.pkl")
    # Shuffle, split and return.
    return split_data(raw_data.sample(frac=1), split_param)
