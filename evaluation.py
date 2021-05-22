import joblib
from data_preprocessing import load_training_data, contsruct_tf_dataset
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import tensorflow as tf

BATCH_SIZE = 300


def prepare_data():

    questions_train_processed, questions_val_processed = pd.Series(), pd.Series()
    with open("./vqa_objects/processed_questions.pkl", "rb") as f:
        questions_train_processed, questions_val_processed = joblib.load(f)

    training_data = load_training_data()

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)

    for train_index, val_index in sss.split(
        training_data["images_train"], training_data["answer_train"]
    ):
        TRAIN_INDEX = train_index
        VAL_INDEX = val_index

    tr_questions = np.array(questions_train_processed)[TRAIN_INDEX.astype(int)]
    val_questions = np.array(questions_train_processed)[VAL_INDEX.astype(int)]

    tr_answers = np.array(training_data["answers_train"])[TRAIN_INDEX.astype(int)]
    val_answers = np.array(training_data["answers_train"])[VAL_INDEX.astype(int)]

    tr_answer = np.array(training_data["answer_train"])[TRAIN_INDEX.astype(int)]
    val_answer = np.array(training_data["answer_train"])[VAL_INDEX.astype(int)]

    tr_images = np.array(training_data["images_train"])[TRAIN_INDEX.astype(int)]
    val_images = np.array(training_data["images_train"])[VAL_INDEX.astype(int)]

    return {
        "tr": {
            "questions": tr_questions,
            "answers": tr_answers,
            "answer": tr_answer,
            "images": tr_images,
        },
        "val": {
            "questions": val_questions,
            "answers": val_answers,
            "answer": val_answer,
            "images": val_images,
        },
    }


def construct_dataset():
    def map_func_eval(img_name, ques):
        img_tensor = np.load(
            "features/" + img_name.decode("utf-8").split("/")[-1][:-4][-6:] + ".npy"
        )
        return img_tensor, ques

    dataset = contsruct_tf_dataset()
    dataset_tr_eval = tf.data.Dataset.from_tensor_slices(
        (dataset["image_list_tr"], dataset["question_tr"])
    )

    # Use map to load the numpy files in parallel
    dataset_tr_eval = dataset_tr_eval.map(
        lambda item1, item2: tf.numpy_function(
            map_func_eval, [item1, item2], [tf.float32, tf.int32]
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    # batch
    dataset_tr_eval = dataset_tr_eval.batch(BATCH_SIZE)

    dataset_vl_eval = tf.data.Dataset.from_tensor_slices(
        (dataset["image_list_vl"], dataset["question_vl"])
    )

    # Use map to load the numpy files in parallel
    dataset_vl_eval = dataset_vl_eval.map(
        lambda item1, item2: tf.numpy_function(
            map_func_eval, [item1, item2], [tf.float32, tf.int32]
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    # batch
    dataset_vl_eval = dataset_vl_eval.batch(BATCH_SIZE)

    return dataset_tr_eval, dataset_vl_eval


def predict_answers(dataset, model, labelencoder):
    """
    Prediction function
    """
    predictions = []
    for img, ques in dataset:
        preds = model([img, ques])
        predictions.extend(preds)

    y_classes = tf.argmax(predictions, axis=1, output_type=tf.int32)
    y_predict = labelencoder.inverse_transform(y_classes)
    return y_predict


def perform_predictions(dataset_tr_eval, dataset_vl_eval, model, labelencoder):
    # predict answers for the train set
    y_predict_text_tr = predict_answers(dataset_tr_eval, model, labelencoder)

    # predict answers for the validation set
    y_predict_text_vl = predict_answers(dataset_vl_eval, model, labelencoder)

    return y_predict_text_tr, y_predict_text_vl


def model_metric(predictions, truths):
    """
    Measures the accuracy of the predictions

    Input:
        predictions : predictions
        truths      : ground truth answers

    Returns:
        Accuracy measure of the model
    """

    total = 0
    correct_val = 0.0

    for prediction, truth in zip(predictions, truths):

        temp_count = 0
        total += 1

        for _truth in truth.split(";"):
            if prediction == _truth:
                temp_count += 1

        # accuracy = min((# humans that provided that answer/3) , 1)
        if temp_count > 2:
            correct_val += 1
        else:
            correct_val += float(temp_count) / 3

    return (correct_val / total) * 100


def main():
    prepared_data = prepare_data()
    label_encoder = load_training_data()["labelencoder"]
    dataset_tr_eval, dataset_vl_eval = construct_dataset()
    y_predict_text_tr, y_predict_text_vl = perform_predictions(
        dataset_tr_eval, dataset_vl_eval, model, label_encoder
    )

    tr_score = model_metric(y_predict_text_tr, prepared_data["tr"]["answers"])

    print("Final Accuracy on the train set is", tr_score)

    val_score = model_metric(y_predict_text_vl, prepared_data["val"]["answers"])

    print("Final Accuracy on the validation set is", val_score)
