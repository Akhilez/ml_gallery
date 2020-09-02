How to execute:

1. If there's no data directory, then run init.sh
2. Install the python dependencies from `pip install -r requirements.txt`
3. Run `python preprocessors/custom.py` for base model. `python preprocessors/bert.py` for BERT
    - This will create 4 .csv files in the data dir.
4. Make sure that the directory `trainers/models` exists. Or create one with `mkdir trainers/models`
5. Run `python trainers/custom_trainer.py` to train the model with the data. If you choose to train BERT, run `python trainer/bert_trainer.py`
    - This will create the models in the `trainers/models` directory.
6. You can predict the labels for your own text by adding your text to `sentences` list in `predictors/make_predictions.py` and run the file `python predictors/make_predictions.py`

---

### Notes

Plan:
1. Do complete preprocessing once and store all the data into a single file.
    - options:
        1. use standard python dir parsing and get the data
        2. Use the flow_from_dir function <---
    - tokenizing options:
        - Use a predefined tokenizer from BERT
        - Create a tokenizer
    - steps:
        - create the tokenizer
        - For each batch:
            1. Clean the data
            2. Add sentences to tokenizer
            2. Create sequences
            3. Append x and y to a file
        - save the tokenizer in json
2. Training
    - options:
        - Use embedding, lstm, linear, conv, attention
    - options:
        - Would you store all of testing data for validation in RAM?
        - Create a generator function that will provide random test data for validation? <---
    - steps:
        - Load the x and y preprocessed data
        - Load tokenizer from json
        - create the model
        - train with live evaluation
            - macro avg f1
        - save model
3. Predicting
    - steps:
        - Load the model and tokenizer
        - load text from some source
        - perform cleanup if any.
        - convert to sequences
        - predict
        - convert predicted one-hot to string labels
            
 
TODO:
 - comment blocs to describe stuff


INTRO

I divided the task into 3 modules.
 -  preprocessing
 -  training
 -  predicting
 
1.  preprocessing will take pure texts from the directories
    and output csv files with integer sequences and one-hot labels.
     - It uses two kinds of tokenizers based on user's specification
        - Bert: for bert model.
        - custom: for my own model.
    This is a one-time module. The generated files can be reused for 
    different training experiments.

2.  Training:
    This module will take the csv files as inputs
    and outputs a trained deep learning model.

    Experiments:
     - Linear
        - Global avg pooling + linear:
            - Gave ok results.
            - Fastest training time
            - 99.99% training accuracy, 76% testing accuracy
            - Over-fitting in the loss graph.
            - Making the model larger negatively effected the performance.
            - Best model considering the accuracy/trainingTime values
            - sequence length: 150
        - sequence len: 512
            - Not better than 150 length
        - Flatten instead of Global Avg Pooling
            - Slightly lower performance than global avg
        - bert tokenizer + global avg pool
            - Just as good as the base model
            - Slower to train due to larger vocab size
     - lstm:
        - Not as good as linear results.
        - Slower to converge + train.
     - Bidirectional lstm:
        - Not as good as lstm
     - 2 layer lstm
        - VERY bad performance.
     - Conv1d + global avg pooling
        - 71%
     - Conv1d + max-pool * 3
        - 60+%
     - BERT
        - We need time, energy and money to run this bad boy.
        - After 7 epochs: Training accuracy 96%, test accuracy: 85%
        - No over-fitting at all
        - Best accuracy so far
     - BERT with only last layer training
        - Took as long as fine-tuning
        - Little to no improvement in loss and accuracy

3.  Predicting:
    This module takes input as follows:
     - Sequencer from preprocessing module
     - the trained model from training module
     - Free text from user [optional]
     - the test dataset from directories. [optional]
    The output of this module is to predict labels for the given text.

references:
 - https://towardsdatascience.com/bert-to-the-rescue-17671379687f
    - https://github.com/shudima/notebooks/blob/master/BERT_to_the_rescue.ipynb
 - Coursera tensorflow in practice
    - https://github.com/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%202%20-%20Lesson%202.ipynb
 - https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
