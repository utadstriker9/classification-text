import pandas as pd
import datetime
import numpy as np
from src.text_preprocessing import generate_preprocessor
import src.utils as utils
from src.model import train_model, get_best_model, get_best_threshold
import json


# Read Data
def read_data(return_file=True):
    existing_data = utils.load_json(CONFIG_DATA["data_set_path"])
    existing_data = existing_data.drop_duplicates(
        subset=CONFIG_DATA["text_column"], keep="first"
    )
    print("Existing data inputted, data shape  :", existing_data.shape)
    new_data = pd.read_excel(CONFIG_DATA["raw_new_dataset_path"])

    # print data
    print("New data inputted, data shape  :", new_data.shape)

    # Remove duplicates data
    new_data = new_data.drop_duplicates(subset=CONFIG_DATA["text_column"], keep="first")
    new_data_exc = new_data[
        ~new_data[CONFIG_DATA["text_column"]].isin(
            existing_data[CONFIG_DATA["text_column"]]
        )
    ]
    data = pd.concat([existing_data, new_data_exc], axis=0, ignore_index=True)

    # Print data
    print("Ready preprocessing new data, data shape   :", new_data_exc.shape)

    # Return data
    if return_file:
        return {"new_data": new_data_exc, "data": data}


if __name__ == "__main__":
    # 1. Load configuration file
    CONFIG_DATA = utils.config_load()

    # 2. Read Data
    rd = read_data()

    # 3. Generate Preprocessor
    data = rd["data"]
    new_data = rd["new_data"]
    data_new = generate_preprocessor(new_data, CONFIG_DATA, return_file=True)

    # Concat to Previous Data Clean
    data_clean = utils.load_json(CONFIG_DATA["data_clean_path"])
    all_data_clean = pd.concat([data_clean, data_new], axis=0, ignore_index=True)

    # Train & Optimize the model
    tm = train_model(all_data_clean, CONFIG_DATA)

    # Get the best model
    list_of_tuned_model = tm["list_of_tuned_model"]
    bm = get_best_model(list_of_tuned_model)

    # Get the best threshold for the best model
    X_test = tm["X_test"]
    y_test = tm["y_test"]
    bt = get_best_threshold(X_test, y_test, bm["best_model"])

    # Get Model Log
    model_record = {
        "timestamp": [utils.time_stamp()],
        "model_name": [bm["model_name"]],
        "model_params": [bm["model_params"]],
        "threshold": [bt["best_threshold"]],
        "test_accuracy_score": [bm["metric_score"]],
        "test_metric_score": [bt["metric_score"]],
    }

    data_model_record = utils.load_json(CONFIG_DATA["bm_record"])
    data_model_record = pd.concat(
        [data_model_record, pd.DataFrame(model_record)], axis=0, ignore_index=True
    )

    # Running Dump File
    # Data
    utils.dump_json(data, CONFIG_DATA["data_set_path"])
    utils.dump_json(all_data_clean, CONFIG_DATA["data_clean_path"])

    # Preprocessing
    utils.pickle_dump(tm["tfidf_vectorizer"], CONFIG_DATA["tfidf_vectorizer_path"])
    utils.pickle_dump(tm["label_encoder"], CONFIG_DATA["label_encoder_path"])
    utils.pickle_dump(tm["X_train"], CONFIG_DATA["train_set_path"][0])
    utils.pickle_dump(tm["X_test"], CONFIG_DATA["test_set_path"][0])
    utils.pickle_dump(tm["y_train"], CONFIG_DATA["train_set_path"][1])
    utils.pickle_dump(tm["y_test"], CONFIG_DATA["test_set_path"][1])

    # Modelling
    utils.pickle_dump(tm["list_of_param"], CONFIG_DATA["list_of_param_path"])
    utils.pickle_dump(tm["list_of_model"], CONFIG_DATA["list_of_model_path"])
    utils.pickle_dump(
        tm["list_of_tuned_model"], CONFIG_DATA["list_of_tuned_model_path"]
    )

    # Best Model
    if (
        data_model_record["test_accuracy_score"].iloc[-1]
        >= data_model_record["test_accuracy_score"].max()
    ):
        utils.pickle_dump(bm["best_model"], CONFIG_DATA["best_model_path"])
        utils.pickle_dump(bt["best_threshold"], CONFIG_DATA["best_threshold_path"])
        utils.dump_json(data_model_record, CONFIG_DATA["bm_record"])
        print("Metric score increased, new model updated")
    else:
        utils.dump_json(data_model_record, CONFIG_DATA["bm_record"])
        print("Metric score not increased, new model not updated")
