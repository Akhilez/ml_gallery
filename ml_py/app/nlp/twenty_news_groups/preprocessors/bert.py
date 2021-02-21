from preprocessors.preprocessor import Preprocessor, BERT

if __name__ == "__main__":
    preprocessor = Preprocessor(BERT)
    preprocessor.make_csv(
        data_input_path="../data/train", data_output_path="../data/train_bert.csv"
    )
    preprocessor.make_csv(
        data_input_path="../data/test", data_output_path="../data/test_bert.csv"
    )
