import csv
from tensorflow.keras.preprocessing import text_dataset_from_directory
from preprocessors.sequencers import CustomSequencer, BertSequencer

BERT = "bert"
CUSTOM = "custom"


class Preprocessor:
    def __init__(self, tokenizer_type=CUSTOM):
        self.tokenizer_type = tokenizer_type
        self.max_char_length = 5000
        self.batch_size = 100
        self.tokenizer_path = "custom_tokenizer.json"

    def make_csv(self, data_input_path, data_output_path, use_saved_tokenizer=False):
        dataset = text_dataset_from_directory(
            data_input_path,
            label_mode="categorical",
            batch_size=self.batch_size,
            max_length=self.max_char_length,
            shuffle=True,
        )
        with open(data_output_path, "w") as data_file:
            writer = csv.writer(data_file)

            if self.tokenizer_type == BERT:
                sequencer = BertSequencer()
                self._prepare_and_write(dataset, sequencer, writer)

            elif self.tokenizer_type == CUSTOM:
                sequencer = CustomSequencer()
                if use_saved_tokenizer:
                    sequencer.tokenizer = CustomSequencer.load_tokenizer(
                        self.tokenizer_path
                    )
                else:
                    for batch in dataset:
                        sentences, labels = batch
                        sentences = [
                            sentence.decode("utf-8") for sentence in sentences.numpy()
                        ]
                        sequencer.feed_tokenizer(sentences)
                    sequencer.save_tokenizer(self.tokenizer_path)
                self._prepare_and_write(dataset, sequencer, writer)

    @staticmethod
    def _prepare_and_write(dataset, sequencer, csv_writer):
        for batch in dataset:
            sentences, labels = batch
            sentences = [sentence.decode("utf-8") for sentence in sentences.numpy()]
            sentences = sequencer.make_sequences(sentences)
            csv_writer.writerows(
                list(sequence) + list(label.numpy().astype(int))
                for sequence, label in zip(sentences, labels)
            )
