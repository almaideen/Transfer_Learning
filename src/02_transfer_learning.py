import argparse
import os
import shutil
from tqdm import tqdm
import logging
from utils.common import read_yaml, create_directories
import random
import tensorflow as tf
import numpy as np


STAGE = "Transfer Learning" ## <<< change stage name

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )

def update_even_odd(list_of_labels):
    for idx,label in enumerate(list_of_labels):
        even_condition = label%2==0
        list_of_labels[idx] = np.where(even_condition, 1, 0)
    return list_of_labels


def main(config_path):
    ## read config files
    config = read_yaml(config_path)

    (X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train_full = X_train_full / 255.0
    X_test = X_test / 255.0
    X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

    y_train_bin,y_test_bin,y_valid_bin = update_even_odd([y_train,y_test,y_valid])

    seed = 2022
    tf.random.set_seed(seed)
    np.random.seed(seed)

    #loading base model
    base_model_path = os.path.join("artifacts","models","base_model.h5")
    base_model = tf.keras.models.load_model(base_model_path)

    #freezing weights for layers except for output layer

    for layer in base_model.layers[:-1]:
        layer.trainable = False

    base_layer = base_model.layers[:-1]
    #defining new model and adding output layer

    new_model = tf.keras.models.Sequential(base_layer)
    new_model.add(
        tf.keras.layers.Dense(2,activation="softmax",name="output_layer")
    )

    LOSS = "sparse_categorical_crossentropy"
    OPTIMIZER = "SGD"
    METRICS = ["accuracy"]

    new_model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)
    new_model.summary()

    #training transfer learning model
    history = new_model.fit(X_train, y_train_bin, epochs=10, validation_data=(X_valid, y_valid_bin), verbose=2)

    #saving the transfer learned model
    model_dir_path = os.path.join("artifacts", "models")
    model_file_path = os.path.join(model_dir_path, "transfer_learned_model.h5")
    new_model.save(model_file_path)

    logging.info(f"Transfer learned model is saved at {model_file_path}")
    logging.info(f"evaluation metrics {new_model.evaluate(X_test, y_test_bin)}")





if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e