from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pytorch_pretrained_bert import BertTokenizer
from abc import ABC, abstractmethod


class Sequencer(ABC):
    @abstractmethod
    def make_sequences(self, sentences):
        pass

    @staticmethod
    def clean(sentence):
        sentence = Sequencer.remove_before_word(sentence, "@")
        sentence = sentence.replace("\\n", " ")
        sentence = sentence.replace("\\", "")
        return sentence

    @staticmethod
    def remove_before_word(sentence, word):
        try:
            subject_index = sentence.index(word)
            sentence = sentence[subject_index + len(word) :]
        except:
            # print(f"HEY! No word '{word}' in {sentence[:30]}...")
            pass
        return sentence


class CustomSequencer(Sequencer):
    def __init__(self):
        self.oov_token = "<OOV>"
        self.vocab_size = 10000
        self.sequence_length = 150

        self.tokenizer = Tokenizer(oov_token=self.oov_token, num_words=self.vocab_size)

    def feed_tokenizer(self, sentences):
        clean_texts = [self.clean(text) for text in sentences]
        self.tokenizer.fit_on_texts(clean_texts)

    def make_sequences(self, sentences):
        clean_texts = [self.clean(text) for text in sentences]
        sequences = self.tokenizer.texts_to_sequences(clean_texts)
        sequences = pad_sequences(
            sequences,
            self.sequence_length,
            padding="post",
            truncating="post",
            dtype="int",
        )
        return sequences

    def save_tokenizer(self, path):
        with open(path, "w") as tokenizer_file:
            tokenizer_file.write(self.tokenizer.to_json())

    @staticmethod
    def load_tokenizer(path):
        with open(path, "r") as tokenizer_file:
            return tokenizer_from_json(tokenizer_file.read())


class BertSequencer(Sequencer):
    def __init__(self):
        self.sequence_length = 510
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )

    def make_sequences(self, sentences):
        clean_texts = [self.clean(text) for text in sentences]
        test_tokens = list(
            map(
                lambda t: ["[CLS]"] + self.tokenizer.tokenize(t)[:510] + ["[SEP]"],
                clean_texts,
            )
        )
        sequences = list(map(self.tokenizer.convert_tokens_to_ids, test_tokens))
        sequences = pad_sequences(
            sequences, maxlen=512, truncating="post", padding="post", dtype="int"
        )
        return sequences
