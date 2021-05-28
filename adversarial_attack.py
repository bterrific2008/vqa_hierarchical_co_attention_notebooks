import joblib
from data_preprocessing import load_training_data, contsruct_tf_dataset
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from random import choice
import tensorflow as tf
import json

BATCH_SIZE = 1
model = tf.keras.models.load_model('hierarchical_vqa_model')
with open("vqa_objects/text_tokenizer.pkl", "rb") as f:
    tok = joblib.load(f)
with open('vqa_objects/labelencoder.pkl', 'rb') as f:
    labelencoder = joblib.load(f)
embeddings = model.get_layer('embedding').get_weights()[0].copy()
encode_embeddings = {i: embeddings[i] for i in range(len(embeddings))}

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
    def map_func_eval(img_name, ques, name):
        img_tensor = np.load(
            "features/" + img_name.decode("utf-8").split("/")[-1][:-4][-6:] + ".npy"
        )
        return img_tensor, ques, name

    dataset = contsruct_tf_dataset()
    dataset_tr_eval = tf.data.Dataset.from_tensor_slices(
        (dataset["image_list_tr"], dataset["question_tr"], dataset["image_list_tr"])
    )

    # Use map to load the numpy files in parallel
    dataset_tr_eval = dataset_tr_eval.map(
        lambda item1, item2, item3: tf.numpy_function(
            map_func_eval, [item1, item2, item3], [tf.float32, tf.int32, tf.string]
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    # batch
    dataset_tr_eval = dataset_tr_eval.batch(BATCH_SIZE)

    dataset_vl_eval = tf.data.Dataset.from_tensor_slices(
        (dataset["image_list_vl"], dataset["question_vl"], dataset["image_list_vl"])
    )

    # Use map to load the numpy files in parallel
    dataset_vl_eval = dataset_vl_eval.map(
        lambda item1, item2, item3: tf.numpy_function(
            map_func_eval, [item1, item2, item3], [tf.float32, tf.int32, tf.string]
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    # batch
    dataset_vl_eval = dataset_vl_eval.batch(BATCH_SIZE)

    return dataset_tr_eval, dataset_vl_eval


def calculate_model_jacobian(img_tensor, question_data_train):
    with tf.GradientTape() as g:
        x = tf.convert_to_tensor(question_data_train, dtype=tf.dtypes.float32)

        image_feat_dense_output = model.get_layer("Image_Feat_Dense")(img_tensor)
        image_feat_dense_dropout_output = model.get_layer("dropout")(
            image_feat_dense_output
        )

        ques_feat_w = model.get_layer("embedding")(x)
        g.watch(ques_feat_w)

        # word level
        Hv_w, Hq_w = model.get_layer("AttentionMaps_Word")(
            image_feat_dense_dropout_output, ques_feat_w
        )
        v_w, q_w = model.get_layer("ContextVector_Word")(
            image_feat_dense_dropout_output, ques_feat_w, Hv_w, Hq_w
        )
        feat_w = model.get_layer("tf.math.add")(v_w, q_w)
        h_w = model.get_layer("h_w_Dense")(feat_w)

        # phrase level
        ques_feat_p = model.get_layer("PhraseLevelFeatures")(ques_feat_w)
        Hv_p, Hq_p = model.get_layer("AttentionMaps_Phrase")(
            image_feat_dense_dropout_output, ques_feat_p
        )
        v_p, q_p = model.get_layer("ContextVector_Phrase")(
            image_feat_dense_dropout_output, ques_feat_p, Hv_p, Hq_p
        )
        feat_p = model.get_layer("tf.math.add_1")(v_p, q_p)
        con_feat_p = model.get_layer("concatenate")([feat_p, h_w])
        h_p = model.get_layer("h_p_Dense")(con_feat_p)

        # sentence level
        lstm_output = model.get_layer("lstm")(phrase)
        Hv_s, Hq_s = model.get_layer("AttentionMaps_Sent")(
            image_feat_dense_dropout_output, lstm_output
        )
        v_s, q_s = model.get_layer("ContextVector_Sent")(
            image_feat_dense_dropout_output, ques_feat_p, Hv_s, Hq_s
        )
        feat_s = model.get_layer("tf.math.add_2")(v_s, q_s)
        con_feat_s = model.get_layer("concatenate_1")([feat_s, h_p])
        h_s = model.get_layer("h_s_Dense")(con_feat_s)
        z = model.get_layer("z_Dense")(h_s)
        dropout_z = model.get_layer("dropout_1")(z)
        result = model.get_layer("dense_12")(dropout_z)

    grads = g.jacobian(result, ques_feat_w)
    return grads, result


def predicted_class(prediction):
    return labelencoder.inverse_transform(
        tf.argmax(
            prediction,
            axis=1,
            output_type=tf.int32,
        )
    )[0]


def adversarial_generation(img, ques, img_name):
    original_prediction = model([img, ques])
    original_predicted_class = predicted_class(original_prediction)

    adversarial_question = ques.copy()
    jacobian, adversarial_prediction = calculate_model_jacobian(
            img, adversarial_question
        )
    adversarial_predicted_class = predicted_class(adversarial_prediction)

    original_question = tok.sequences_to_texts(ques)[0]
    generation_data = {
        'original_prediction': original_prediction,
        'original_img': img_name,
        'original_question': original_question
    }
    generation_log = []
    generation_step = 0
    
    while adversarial_predicted_class == original_predicted_class:
        generation_log_step = {'step': generation_step}
        
        _, word_indicies = np.where(adversarial_question == 0)
        word_idx = choice(word_indicies)
        generation_log_step['word_idx'] = word_idx
        
        weight_shift = []
        for word in tok.word_index.values():
            weight_shift.append(
                    (
                        np.linalg.norm(
                            np.sign(
                                encode_embeddings[adversarial_question[0][word_idx]] - encode_embeddings[word]
                            ) - np.sign(
                                jacobian[0][word_idx]
                            )
                        ),
                        word,
                    )
                )

        _, update_word = min(weight_shift)

        generation_log_step['top_10'] = list(sorted(weight_shift)[:10])
        generation_log_step['bottom_10'] = list(sorted(weight_shift, reverse=True)[:10])
        generation_log_step['update_word'] = update_word
        print("update word: ", update_word)
        
        adversarial_question[0][word_idx] = update_word
        jacobian, adversarial_prediction = calculate_model_jacobian(
            img, adversarial_question
        )
        adversarial_predicted_class = predicted_class(adversarial_prediction)

        generation_log_step['adverasrial_predicted_class'] = adversarial_predicted_class
        generation_log_step['adversarial_question'] = adversarial_question
        
        generation_log.append(generation_log_step)
        generation_step += 1
        print(generation_step, adversarial_predicted_class, original_predicted_class)

    generation_data['generation_log'] = generation_log
    return adversarial_question, generation_data


def create_adversarial_prompts(dataset):
    """
    Mass generation function
    """
    logs = []
    for img, ques, img_name in dataset:
        _, generation_log = adversarial_generation(img, ques, img_name)
        logs.append(generation_log)
    
    return generation_log


def perform_predictions(dataset_tr_eval, dataset_vl_eval):
    # predict answers for the train set
    y_adversarial_tr = create_adversarial_prompts(dataset_tr_eval)

    # predict answers for the validation set
    y_adversarial_vl = create_adversarial_prompts(dataset_vl_eval)

    return y_adversarial_tr, y_adversarial_vl


def main():
    dataset_tr_eval, dataset_vl_eval = construct_dataset()
    y_adversarial_tr, y_adversarial_vl = perform_predictions(
        dataset_tr_eval, dataset_vl_eval
    )

    with open('y_adversarial_tr.json', 'w', encoding='utf-8') as f:
        json.dump(y_adversarial_tr, f, ensure_ascii=False, indent=4)

    with open('y_adversarial_vl.json', 'w', encoding='utf-8') as f:
        json.dump(y_adversarial_vl, f, ensure_ascii=False, indent=4)