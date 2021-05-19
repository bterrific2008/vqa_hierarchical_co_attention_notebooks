import pickle

import joblib
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras import utils
from tensorflow.keras.preprocessing import text

BATCH_SIZE = 300
BUFFER_SIZE = 5000


def load_training_data():
    print("Load Questions and Image Training Data")
    with open("preprocessed_data/vqa_raw_train2014_top1000.json", "rb") as f:
        questions_train, answer_train, answers_train, images_train = joblib.load(f)

    with open("preprocessed_data/vqa_raw_val2014_top1000.json", "rb") as f:
        questions_val, answer_val, answers_val, images_val = joblib.load(f)

    print("Load Text Tokenizers and Encoders")
    tok = text.Tokenizer(filters="")
    # load from disk
    with open("vqa_objects/text_tokenizer.pkl", "rb") as f:
        tok = joblib.load(f)

    # load from disk
    with open("vqa_objects/tokenised_data_post.pkl", mode="rb") as f:
        question_data_train, question_data_val = pickle.load(f)

    # load from disk
    with open("vqa_objects/labelencoder.pkl", "rb") as f:
        labelencoder = joblib.load(f)

    return {
        "questions_train": questions_train,
        "answer_train": answer_train,
        "answers_train": answers_train,
        "images_train": images_train,
        "questions_val": questions_val,
        "answer_val": answer_val,
        "answers_val": answers_val,
        "images_val": images_val,
        "tok": tok,
        "question_data_train": question_data_train,
        "question_data_val": question_data_val,
        "labelencoder": labelencoder,
    }


def data_matrices(images_train, answer_train, question_data_train, labelencoder):
    # Prepare data matrices
    print("Prepare data matrices")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)

    for train_index, val_index in sss.split(images_train, answer_train):
        TRAIN_INDEX = train_index
        VAL_INDEX = val_index

    # image data
    image_list_tr, image_list_vl = (
        np.array(images_train)[TRAIN_INDEX.astype(int)],
        np.array(images_train)[VAL_INDEX.astype(int)],
    )

    # question data
    question_tr, question_vl = (
        question_data_train[TRAIN_INDEX],
        question_data_train[VAL_INDEX],
    )

    def get_answers_matrix(answers, encoder):
        """
        One-hot-encodes the answers

        Input:
            answers:	list of answer
            encoder:	a scikit-learn LabelEncoder object

        Output:
            A numpy array of shape (# of answers, # of class)
        """
        y = encoder.transform(answers)  # string to numerical class
        nb_classes = encoder.classes_.shape[0]
        Y = utils.to_categorical(y, nb_classes)
        return Y

    # answer data
    answer_matrix = get_answers_matrix(answer_train, labelencoder)
    answer_tr, answer_vl = answer_matrix[TRAIN_INDEX], answer_matrix[VAL_INDEX]

    return image_list_tr, image_list_vl, question_tr, question_vl, answer_tr, answer_vl


def tf_dataset(
    image_list_tr, question_tr, answer_tr, image_list_vl, question_vl, answer_vl
):
    print("Create TF Dataset")

    # loading the numpy files
    def map_func(img_name, ques, ans):
        img_tensor = np.load(
            "features/" + img_name.decode("utf-8").split(".")[0][-6:] + ".npy"
        )
        return img_tensor, ques, ans

    dataset_tr = tf.data.Dataset.from_tensor_slices(
        (image_list_tr, question_tr, answer_tr)
    )

    # Use map to load the numpy files in parallel
    dataset_tr = dataset_tr.map(
        lambda item1, item2, item3: tf.numpy_function(
            map_func, [item1, item2, item3], [tf.float32, tf.int32, tf.float32]
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    # Shuffle and batch
    dataset_tr = dataset_tr.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    dataset_tr = dataset_tr.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    dataset_vl = tf.data.Dataset.from_tensor_slices(
        (image_list_vl, question_vl, answer_vl)
    )

    # Use map to load the numpy files in parallel
    dataset_vl = dataset_vl.map(
        lambda item1, item2, item3: tf.numpy_function(
            map_func, [item1, item2, item3], [tf.float32, tf.int32, tf.float32]
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    # Shuffle and batch
    dataset_vl = dataset_vl.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    dataset_vl = dataset_vl.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset_tr, dataset_vl


def construct_data_matrices(training_data):
    (
        image_list_tr,
        image_list_vl,
        question_tr,
        question_vl,
        answer_tr,
        answer_vl,
    ) = data_matrices(
        training_data["images_train"],
        training_data["answer_train"],
        training_data["question_data_train"],
        training_data["labelencoder"],
    )

    return {
        "image_list_tr": image_list_tr,
        "image_list_vl": image_list_vl,
        "question_tr": question_tr,
        "question_vl": question_vl,
        "answer_tr": answer_tr,
        "answer_vl": answer_vl,
    }


def contsruct_tf_dataset():
    training_data = load_training_data()
    data_matrices = construct_data_matrices(training_data)
    dataset_tr, dataset_vl = tf_dataset(
        data_matrices["image_list_tr"],
        data_matrices["question_tr"],
        data_matrices["answer_tr"],
        data_matrices["image_list_vl"],
        data_matrices["question_vl"],
        data_matrices["answer_vl"],
    )

    return {
        "dataset_tr": dataset_tr,
        "dataset_vl": dataset_vl,
    }


if __name__ == "__main__":
    tf_dataset = contsruct_tf_dataset()
    print("dataset_tr", tf_dataset["dataset_tr"].shape)
    print("dataset_vl", tf_dataset["dataset_vl"].shape)