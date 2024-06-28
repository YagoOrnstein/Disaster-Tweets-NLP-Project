import pandas as pd
import tensorflow as tf
import helper_functions as hlpf
import numpy as np
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import TextVectorization

# For making the dataframes better
desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 10)

"""
NLP has the goal of deriving information out of natural language (could be sequences text or speach)
Another common term for NLP problems is sequence to sequence problems (seq2seq)
"""

train_df = pd.read_csv(
    r"C:\Users\yazo_\OneDrive\Masaüstü\Tensorflow Projects\NLP\Natural Language Processing with Disaster Tweets\train.csv")

test_df = pd.read_csv(
    r"C:\Users\yazo_\OneDrive\Masaüstü\Tensorflow Projects\NLP\Natural Language Processing with Disaster Tweets\test.csv")

logs_dir = r"C:\Users\yazo_\OneDrive\Masaüstü\Tensorflow Projects\NLP\logs"

# Visualize the data
train_df.head(10)

print(list(train_df.columns.values))
# Our headers are ['id', 'keyword', 'location', 'text', 'target']

print(train_df["text"][0])
# So will try to build a model to use the text data to predict target

# Shuffle training dataframe
train_df_shuffled = train_df.sample(frac=1, random_state=42)
train_df_shuffled.head(2)

# Check the test data
test_df.head()

# How many examples of each class?
train_df.target.value_counts()

# How mant total samples?
len(train_df), len(test_df)

# Let's visualize some random training examples
random_index = random.randint(0, len(train_df) - 5)
for row in train_df_shuffled[["text", "target"]][random_index:random_index + 5].itertuples():
    _, text, target = row
    print(f"Target: {target}", "(real disaster)" if target > 0 else "(not real disaster)")
    print(f"Text \n{text}\n")
    print("------\n")

# Splitting data into training and validation datasets
# We will use sklearn.model_selection ==> train_test_split
X = train_df_shuffled["text"].to_numpy()
y = train_df_shuffled["target"].to_numpy()

# Creating a validation data with 10% of the training data
train_sentences, test_sentences, train_labels, test_labels = train_test_split(X, y, test_size=0.1, random_state=42)

len(train_sentences), len(train_labels), len(test_sentences), len(test_labels)

# Check 10 examples form samples
print(train_sentences[:10], train_labels[:10])

# Converting text data to numbers using tokenization and embeddings
"""
When dealing with text problem, one of the first thing you'll have to do before you can
build a model is to convert your text to numbers

There are a few ways to do this:
* Tokenization - direct mapping of token (a token could be a word or a character) to number
* Embedding - create a matrix of feature vector for each token (the size of the feature vector
can be defined and this embedding can be learned)
"""

# Text vectorization (**from tensorflow.keras.layers import TextVectorization**)
text_vectorizer_standard = TextVectorization(max_tokens=None,  # How many words in vocabulary
                                             standardize="lower_and_strip_punctuation",
                                             split="whitespace",
                                             ngrams=None,  # Create groups of n amount words
                                             output_mode="int",  # How to map tokens to numbers
                                             output_sequence_length=None,  # How long do you want your sequences to be
                                             )

train_sentences[0].split()

# Find the average number of tokens (words) in the training tweets
round(sum([len(i.split()) for i in train_sentences]) / len(train_sentences))

# Setup text vectorization variables
max_vocab_length = 10000  # Max number of words to have in our vocabulary
max_length = 15  # Max length our sequences will be

text_vectorizer = TextVectorization(max_tokens=max_vocab_length,
                                    output_mode="int",
                                    output_sequence_length=max_length)

# Fit the text vectorizer to the training text
text_vectorizer.adapt(train_sentences)

# Create a sample sentence and tokenize it
sample_sentence = "There's a flood in my street!"
text_vectorizer([sample_sentence])

# Choose a random sentence from the training dataset
random_sentence = random.choice(train_sentences)
print(f"Original Text:\n{random_sentence}\n\nVectorized Version:\n{text_vectorizer([random_sentence])}")

# Get the unique words in the vocabulary
words_in_vocab = text_vectorizer.get_vocabulary()  # Get all the unique words in our text data
top_5_words = words_in_vocab[:5]  # Get the most common words
bottom_5_words = words_in_vocab[-5:]
print(f"Number of the words in vocab: {len(words_in_vocab)}")
print(f"5 Most Common Words:{top_5_words}")
print(f"5 Least Common Words:{bottom_5_words}")

# Creating an Embedding using an Embedding Layer
"""
The parameters we care most about for our embedding layer:
* input_dim = the size of the vocabulary

* output_dim = the size of the output embedding vector, for example, a value
100 would mean each token get represented by a vector 100 long

* input_length = length of sequences being passed to embedding layer
"""

embedding = tf.keras.layers.Embedding(input_dim=max_vocab_length,
                                      output_dim=128,
                                      input_length=max_length)

# Get a random sentence from our training set
random_sentence2 = random.choice(train_sentences)

# Embed the random sentence (turn it into dense vectors of fixed size)
sample_embed = embedding(text_vectorizer([random_sentence2]))

print(f"Original Text:\n{random_sentence2}\n\nEmbedded Version:\n{sample_embed}")

# Check out single token's embedding
print(sample_embed[0][0], "\n", sample_embed[0][0].shape, "\n", random_sentence2)

# Modelling a text dataset
"""
Experiments gonna do are:
1- Naive Bayer with TF-IDF encoder
2- Feed-Forward Neural Network (Dense Model)
3- LSTM (RNN)
4- GRU (RNN)
5- Bidirectional-LSTM (RNN)
8- 1D Convolutional Neural Network
9- Pretrained Feature Extractor
10-Pretrained Feature Extractor (10% of data)

* Create a model
* Build a model
* Fit a model
* Evaluate our model
"""

# Create Model 1 (Naive Bayer with TF-IDF encoder from sklearn)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Create tokenization and modelling pipeline
model_1 = Pipeline([
    ("tfidf", TfidfVectorizer()),  # Converts words to numbers using tfidf
    ("clf", MultinomialNB())  # Model the text (CLF is shortcut for Classifier)
])

# Fit the pipeline to the training data
model_1.fit(train_sentences, train_labels)

# Evaluate our baseline model
baseline_score = model_1.score(test_sentences, test_labels)
print(f"Our baseline model achives an accuracy of: {baseline_score * 100:.2f}½")

# Make Predictions
baseline_preds = model_1.predict(test_sentences)
print(baseline_preds[:10])

# Creating an evaluation function for our model experiments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def calculate_results(y_true, y_preds):
    """
    Calculate model's
    * Accuracy (Higher is better)
    * Precision (Higher precision leads to less false positives)
    * Recall (Higher recall leads to less false negatives)
    * F1-score (Combination of precision and recall usually good overall metric for a classification model)
    for a binary classification model.
    """
    # Calculate model accuracy
    model_accuracy = accuracy_score(y_true, y_preds) * 100

    # Calculate model precision, recall and f1-score using "weighted" average
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_preds, average="weighted")
    model_results = {
        "accuracy": model_accuracy,
        "precision": model_precision,
        "recall": model_recall,
        "f1": model_f1
    }
    return model_results


baseline_results = calculate_results(y_true=test_labels,
                                     y_preds=baseline_preds)
print(baseline_results)

# Create Model 2 (Feed-Forward Neural Network (Dense Model))

tf.random.set_seed(42)
model_2_embedding = tf.keras.layers.Embedding(input_dim=max_vocab_length,
                                              output_dim=128,
                                              embeddings_initializer="uniform",
                                              input_length=max_length,
                                              name="embedding_2")

# Create a tensorboard callback ( need to create a new one for each model)
tensorboard_callback_model_0 = hlpf.create_tensorboard_callback(
    dir_name=logs_dir,
    experiment_name="Feed-Forward Neural Network (Dense Model)")

# Build model with the Functional API
inputs = tf.keras.layers.Input(shape=(1,), dtype=tf.string)  # Inputs are 1-dim strings
x = text_vectorizer(inputs)  # Turn the input text into numbers
x = model_2_embedding(x)  # Turn the numbers into embeddings
x = tf.keras.layers.GlobalAveragePooling1D()(x)  # lower the dimensionality of the embedding
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)  # Create the output layer
model_2 = tf.keras.Model(inputs, outputs, name="Model_2_Dense")  # Construct the model

model_2.summary()

# Compile model
model_2.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

# Fit the model
model_2.history = model_2.fit(x=train_sentences,
                              y=train_labels,
                              validation_data=(test_sentences, test_labels),
                              epochs=5,
                              callbacks=[tensorboard_callback_model_0], verbose=0)

# Check the results
model_2.evaluate(test_sentences, test_labels)

# Make some predictions and evaluate those
model_2_pred_probs = model_2.predict(test_sentences)
print(model_2_pred_probs[:5])

# Convert model prediction probabilities to label form
model_2_pred = tf.squeeze(tf.round(model_2_pred_probs))
print(model_2_pred[:10])

# Calculate our Model 2 results
model_2_results = calculate_results(y_true=test_labels,
                                    y_preds=model_2_pred)
print(model_2_results)

print(np.array(list(model_2_results.values())) > np.array(list(baseline_results.values())))

# Visualization of learned embeddings
words_in_vocab = text_vectorizer.get_vocabulary()
len(words_in_vocab), words_in_vocab[:10]

# Model 2 summery
model_2.summary()

# Get the weight matrix of embedding layer
# These are the numerical representations of each token in our training data, which have been learned for 5 epochs
embed_weights = model_2.get_layer("embedding_2").get_weights()[0]
print(embed_weights.shape)

# Model 3 (LSTM (RNN))
"""
RNN is useful for sequence data.
The premise of recurrent neural network is to use the representation 
of a previous input to aid the representation 

LSTM = long short term memory

Structure:
Input(text) -> Tokenize -> Embedding -> Layers(RNN/Dense -> Output(label)
"""
tf.random.set_seed(42)
model_3_embedding = tf.keras.layers.Embedding(input_dim=max_vocab_length,
                                              output_dim=128,
                                              embeddings_initializer="uniform",
                                              input_length=max_length,
                                              name="embedding_2")

inputs = tf.keras.layers.Input(shape=(1,), dtype="string")
x = text_vectorizer(inputs)
x = model_3_embedding(x)
x = tf.keras.layers.LSTM(64, return_sequences=True)(
    x)  # When you're stacking RNN cells together you need to set return_sequences=True
x = tf.keras.layers.LSTM(64)(x)
x = tf.keras.layers.Dense(64, activation="relu")(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model_3 = tf.keras.Model(inputs, outputs, name="Model_3_LSTM")

model_3.summary()

model_3.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

model_3.fit(test_sentences,
            test_labels,
            validation_data=[train_sentences, train_labels],
            epochs=5,
            callbacks=[hlpf.create_tensorboard_callback(
                dir_name=logs_dir,
                experiment_name="LSTM Model 3")], verbose=0)

# Make predictions on the validation dataset
model_3_pred_probs = model_2.predict(test_sentences)
print(model_3_pred_probs.shape, model_3_pred_probs[:10])  # view the first 10

# Round out predictions and reduce to 1-dimensional array
model_3_preds = tf.squeeze(tf.round(model_3_pred_probs))
print(model_3_preds.shape)
print(model_3_preds[:10])

# Calculate LSTM model results
model_3_results = calculate_results(y_true=test_labels,
                                    y_preds=model_3_preds)
print(model_3_results["precision"])


# Compare model 3 to baseline
def compare_results(first_model, second_model, first_model_result, second_model_result):
    """
    first_model: Just name of the first model
    second_model: Name of the second model
    first_model_result: The results which calculated with (calculate_results) function which we defined before
    second_model_result: The results which calculated with (calculate_results) function which we defined before
    """
    print(
        f"{first_model} accuracy: {int(first_model_result['accuracy']):.2f}, {second_model} accuracy: {int(second_model_result['accuracy']):.2f}, Difference: {(int(first_model_result['accuracy']) - int(second_model_result['accuracy'])):.2f}")
    print(

        f"{first_model} precision: {first_model_result['precision']:.2f}, {second_model} precision: {second_model_result['precision']:.2f}, Difference: {(first_model_result['precision'] - second_model_result['precision']):.2f}")
    print(
        f"{first_model} recall: {first_model_result['recall']:.2f}, {second_model} recall: {second_model_result['recall']:.2f}, Difference: {(first_model_result['recall'] - second_model_result['recall']):.2f}")

    print(
        f"{first_model} f1: {first_model_result['f1']:.2f}, {second_model} f1: {second_model_result['f1']:.2f}, Difference: {(first_model_result['f1'] - second_model_result['f1']):.2f}")


compare_results(first_model="Baseline Model", second_model="Model 3 LSTM", first_model_result=baseline_results,
                second_model_result=model_3_results)

# Model 4 (GRU (RNN)
"""
The GRU cell has similar features to an LSTM cell but has less parameters.
tensorflow.keras.layers.GRU()

Architecture
Input (text) -> Tokenize -> Embedding -> Layers -> Output (label probability)
"""

model_4_embedding = tf.keras.layers.Embedding(input_dim=max_vocab_length,
                                              output_dim=128,
                                              embeddings_initializer="uniform",
                                              input_length=max_length,
                                              name="embedding_4")

# Build an RNN using the GRU cell
inputs = tf.keras.layers.Input(shape=(1,), dtype="string")
x = text_vectorizer(inputs)
x = model_4_embedding(x)
# x = tf.keras.layers.GRU(64, return_sequences=True) #  stacking recurrent cells requires return_sequences=True
x = tf.keras.layers.GRU(64)(x)
# x = tf.keras.layers.Dense(64, activation="relu")(x)
# x = tf.keras.layers.GlobalAveragePooling1D()(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model_4 = tf.keras.Model(inputs, outputs, name="model_4_GRU")

model_4.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

model_4_history = model_4.fit(train_sentences,
                              train_labels,
                              epochs=5,
                              validation_data=[test_sentences, test_labels],
                              callbacks=hlpf.create_tensorboard_callback(
                                  dir_name=logs_dir,
                                  experiment_name="GRU Model"), verbose=0)

model_4_pred_probs = model_4.predict(test_sentences)
print(model_4_pred_probs.shape, model_4_pred_probs[:10])

model_4_preds = tf.squeeze(tf.round(model_4_pred_probs))
print(model_4_preds[:10])

model_4_results = calculate_results(y_true=test_labels, y_preds=model_4_preds)

compare_results(first_model="Baseline Model", second_model="GRU Model 4", first_model_result=baseline_results,
                second_model_result=model_4_results)

# Model 5 Bidirectional-LSTM (RNN)
"""
A standard RNN will process a sequence from left to right, where as a bidirectional RNN will process the sequence from 
left to right and then again from right to left.

Intuitively, this can be thought of as if you were reading a sentence for the first time in the normal fashion 
(left to right) but for some reason it didn't make sense so you traverse back 
through the words and go back over them again (right to left).
"""

tf.random.set_seed(42)
embedding_5 = tf.keras.layers.Embedding(input_dim=max_vocab_length,
                                        output_dim=128,
                                        input_length=max_length,
                                        embeddings_initializer="uniform",
                                        name="embedding_5")

model_5_callback = hlpf.create_tensorboard_callback(
    dir_name=logs_dir,
    experiment_name="Bidirectional-LSTM (RNN)")

# Build the model
inputs = tf.keras.Input(shape=(1,), dtype="string")
x = text_vectorizer(inputs)
x = embedding_5(x)
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model_5 = tf.keras.Model(inputs, outputs, name="model_5_bidirectional")

model_5.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

model_5_history = model_5.fit(train_sentences,
                              train_labels,
                              validation_data=[test_sentences, test_labels],
                              epochs=5,
                              callbacks=[model_5_callback],
                              verbose=0)

model_5_preds_probs = model_5.predict(test_sentences)
print(model_5_preds_probs[:5])

model_5_preds = tf.squeeze(tf.round(model_5_preds_probs))
print(model_5_preds[:5])

model_5_result = calculate_results(test_labels, model_5_preds)

compare_results(first_model="Baseline Model", second_model="Bidirectional Model 5", first_model_result=baseline_results,
                second_model_result=model_5_result)

# Model 6 (1D Convolutional Neural Network)

# Example of 1D convolutional neural network
embedding_test = embedding(text_vectorizer(["this is a test sentence"]))  # Turn target sentence into embedding
conv_1d = tf.keras.layers.Conv1D(filters=32,
                                 kernel_size=5,
                                 padding="same",
                                 activation="relu")  # Convolve over the target sequence 5 words at a time
conv_1d_output = conv_1d(embedding_test)  # Pass embedding through 1D convolutional layer
max_pool = tf.keras.layers.GlobalMaxPool1D()
max_pool_output = max_pool(conv_1d_output)  # Get the most important features
print(embedding_test.shape, conv_1d_output.shape, max_pool_output.shape)

"""
The embedding has an output shape dimension of the parameters we set it to (`input_length=15` and `output_dim=128`).

The 1-dimensional convolutional layer has an output which has been compressed inline with its parameters. 
And the same goes for the max pooling layer output.

Our text starts out as a string but gets converted to a feature vector of length 64 through various transformation 
steps (from tokenization to embedding to 1-dimensional convolution to max pool)
"""

print(embedding_test[:1], conv_1d_output[:1], max_pool_output[:1])

tf.random.set_seed(42)
model_6_embedding = tf.keras.layers.Embedding(input_dim=max_vocab_length,
                                              output_dim=128,
                                              embeddings_initializer="uniform",
                                              input_length=max_length,
                                              name="embedding_6")

# Create 1-dimensional convolutional layer to model sequences
inputs = tf.keras.layers.Input(shape=(1,), dtype="string")
x = text_vectorizer(inputs)
x = model_6_embedding(x)
x = tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation="relu", padding="same", strides=1)(x)
x = tf.keras.layers.GlobalMaxPool1D()(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model_6 = tf.keras.Model(inputs, outputs, name="model_6_conv1d")

model_6.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics="accuracy")

model_6.summary()

model_6_history = model_6.fit(train_sentences,
                              train_labels,
                              epochs=5,
                              validation_data=[test_sentences, test_labels],
                              callbacks=[hlpf.create_tensorboard_callback(
                                  dir_name=logs_dir,
                                  experiment_name="Conv1D")])

# Make predictions with model_6
model_6_pred_probs = model_6.predict(test_sentences)
print(model_6_pred_probs[:5])

model_6_preds = tf.squeeze(tf.round(model_6_pred_probs))
print(model_6_preds[:5])

model_6_results = calculate_results(y_true=test_labels,
                                    y_preds=model_6_preds)
print(model_6_results)

# Comparing the results with Baseline Model
compare_results(first_model_result=baseline_results,
                second_model="Conv1D",
                first_model="Baseline Model",
                second_model_result=model_6_results)

# Model 7 Pretrained Feature Extractor
import tensorflow_hub as hub

embed = hub.load(
    "https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/universal-sentence-encoder/2")
embeddings = embed([
    "The quick brown fox jumps over the lazy dog.",
    "I am a sentence for which I would like to get its embedding"])

print(embeddings)

print(embeddings[0][:50])

print(embeddings[0].shape)

# We can use this encoding layer in place of our text_vectorizer and embedding layer
sentence_encoder_layer = hub.KerasLayer(handle=embed,
                                        input_shape=[],  # shape of inputs coming to our model
                                        dtype=tf.string,  # data type of inputs coming to the USE layer
                                        trainable=False,
                                        # keep the pretrained weights (we'll create a feature extractor)
                                        name="USE")

# Create model using the Sequential API
model_7 = tf.keras.Sequential([
    sentence_encoder_layer,  # take in sentences and then encode them into an embedding
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
], name="model_7_USE")

# Compile model
model_7.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

model_7.summary()

model_7_history = model_7.fit(train_sentences,
                              train_labels,
                              validation_data=[test_sentences, test_labels],
                              epochs=5,
                              callbacks=[hlpf.create_tensorboard_callback(
                                  dir_name=logs_dir,
                                  experiment_name="Pretrained Feature Extractor")])

model_7_pred_probs = model_7.predict(test_sentences)
print(model_7_pred_probs[:2])

model_7_preds = tf.squeeze(tf.round(model_7_pred_probs))
print(model_7_preds[:2])

model_7_results = calculate_results(test_labels, model_7_preds)
print(model_7_results)

# Comparing the results with Baseline Model
compare_results(first_model_result=baseline_results,
                second_model="Pretrained Feature Extractor",
                first_model="Baseline Model",
                second_model_result=model_7_results)

# Model 8 Pretrained Feature Extractor (10% of data)
train_sentences_90, test_sentences_10, train_labels_90, test_labels_10 = train_test_split(np.array(train_sentences),
                                                                                          train_labels,
                                                                                          test_size=0.1,
                                                                                          random_state=42)

# Check length of 10 percent datasets
print(f"Total training examples: {len(train_sentences)}")
print(f"Length of 10% training examples: {len(train_sentences_90)}")

# Clone model_6 but reset weights
model_8 = tf.keras.Sequential([
    sentence_encoder_layer,  # take in sentences and then encode them into an embedding
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
], name="model_8_USE")

# Compile model
model_8.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

# Get a summary (will be same as model_6)
model_8.summary()

# Fit the model to 10% of the training data
model_8_history = model_7.fit(x=train_sentences_90,
                              y=train_labels_90,
                              epochs=5,
                              validation_data=(test_sentences, test_labels),
                              callbacks=[hlpf.create_tensorboard_callback(
                                  dir_name=logs_dir,
                                  experiment_name="Pretrained Feature Extractor 10 Percent")])

model_8_pred_probs = model_8.predict(test_sentences)
print(model_8_pred_probs[:2])

model_8_preds = tf.squeeze(tf.round(model_8_pred_probs))
print(model_8_preds[:2])

model_8_results = calculate_results(test_labels, model_8_preds)
print(model_8_results)

compare_results(first_model_result=baseline_results,
                second_model="Pretrained Feature Extractor 10%",
                first_model="Baseline Model",
                second_model_result=model_8_results)

# Combine model results into a DataFrame
all_model_result = pd.DataFrame({"Baseline Model": baseline_results,
                                 "Dense Model": model_2_results,
                                 "LSTM Model": model_3_results,
                                 "GRU Model": model_4_results,
                                 "Bidirectional-LSTM Model": model_5_result,
                                 "1D Convolutional Model": model_6_results,
                                 "Pretrained Model": model_7_results,
                                 "10% Pretrained Model": model_8_results})

all_model_result = all_model_result.transpose()

print(all_model_result)

# Reduce the accuracy to same scale as other metrics
all_model_result["accuracy"] = all_model_result["accuracy"] / 100

# Plot and compare all of the model results
all_model_result.plot(kind="bar", figsize=(10, 7)).legend(bbox_to_anchor=(1.0, 1.0))

# Sort model results by f1-score
all_model_result.sort_values("f1", ascending=False)["f1"].plot(kind="bar", figsize=(10, 7))

# Finding the most wrong examples
val_df = pd.DataFrame({"text": test_sentences,
                       "target": test_labels,
                       "pred": model_7_preds,
                       "pred_prob": tf.squeeze(model_7_pred_probs)})

val_df.head()

# Find the wrong predictions and sort by prediction probabilities
most_wrong = val_df[val_df["target"] != val_df["pred"]].sort_values("pred_prob", ascending=False)
most_wrong.head(10)
most_wrong.tail(10)

# Check the false positives
for row in most_wrong[:5].itertuples():
    _, text, target, pred, pred_prob = row
    print(f"Target: {target}, Pred: {pred}, Prob: {pred_prob}")
    print(f"Text:\n{text}\n")
    print("----------------------\n")

# Check the false negatives
for row in most_wrong[-5:].itertuples():
    _, text, target, pred, pred_prob = row
    print(f"Target: {target}, Pred: {pred}, Prob: {pred_prob}")
    print(f"Text:\n{text}\n")
    print("----------------------\n")

# The Speed/Score Tradeoff
import time


def pred_timer(model, samples):
    """
    Times how long a model takes to make predictions on samples
    """
    start_time = time.perf_counter()  # get start time
    model.predict(samples)
    end_time = time.perf_counter()
    total_time = end_time - start_time
    time_per_pred = total_time / len(samples)
    return total_time, time_per_pred


model_7_total_pred_time, model_7_time_per_pred = pred_timer(model=model_7, samples=test_sentences)

print("Total Time of Pretrained Model:", model_7_total_pred_time, "\n", "Time For Per Pred of Pretrained model:",
      model_7_time_per_pred)

model_1_total_pred_time, model_1_time_per_pred = pred_timer(model=model_1, samples=test_sentences)

print("Total Time of Baseline Model:", model_1_total_pred_time, "\n", "Time For Per Pred of Baseline Model:",
      model_1_time_per_pred)

"""
Baseline model is significantly more faster then pretrained model but pretrained model has little 
bit more accurate and has higher F1 score.
"""
