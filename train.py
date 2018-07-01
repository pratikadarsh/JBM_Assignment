import tensorflow as tf
import pandas as pd
import logging
import estimator
import data_loader

def run(args):
    """ Trains a CNN and dumps the model. """
	
    logging.info("loading data\n")
    # Load Data.
    data = data_loader.load_data((0.9, 0.1, 0))
    logging.info("data loaded\n")
    # Get the model function.
    model = tf.estimator.Estimator(model_fn=estimator.model_fn, model_dir='./model/third',
        params=args)
    logging.debug("model created\n")
    # Model Training.
    input_fn = tf.estimator.inputs.pandas_input_fn(
        x=data["train_X"], y=pd.Series(data["train_Y"]), batch_size=args.batch_size,
        num_epochs=5, shuffle=args.shuffle)
    logging.debug("input_fn created\n")
    logging.info("training model\n")
    model.train(input_fn, steps=args.num_steps)   
    logging.info("training completed, now testing the model\n")
    # Model Testing.
    input_fn = tf.estimator.inputs.pandas_input_fn(
	    x=data["test_X"], y=data["test_Y"], batch_size=1, shuffle=args.shuffle)
    e = model.evaluate(input_fn)
    logging.info("testing completed\n")
    print("Testing Accuracy: ", e['accuracy'])
	
