import tensorflow as tf
import logging
import argparse
import train
import predict

def get_parser():
    """ CLI Argument Parser."""

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, help="train/predict")
    parser.add_argument("--image", type=str)
    parser.add_argument("--arch", default="nn", type=str,
        help="the ML model to be used.")
    parser.add_argument("--batch_size", default=1, type=int,
        help="size of the mini batch")
    parser.add_argument("--learning_rate", default=1E-3, type=float,
        help="learning rate")
    parser.add_argument("--num_steps", default=1000, type=int,
        help="number of steps")
    parser.add_argument("--shuffle", default=False, type=bool,
        help="flag to shuffle the dataset before training")
    return parser

def main():

    logging.basicConfig(filename='classifier.log',
        filemode='w', level=logging.DEBUG)
    tf.logging.set_verbosity(tf.logging.INFO)

    logging.info("reading arguments\n")
    # Parse the CLI Arguments.
    parser = get_parser()
    args = parser.parse_args()

    if args.mode == 'train':
        train.run(args)
    elif args.mode == 'predict':
        predict.run(args)
    else:
        logging.error("Please enter the correct mode of operation.\n")

if __name__ == '__main__':
    main()
