from preprocessors.preprocessor import Preprocessor, CUSTOM

if __name__ == "__main__":
    preprocessor = Preprocessor(CUSTOM)
    preprocessor.make_csv(
        data_input_path="../data/train", data_output_path="../data/train_custom.csv"
    )
    preprocessor.make_csv(
        data_input_path="../data/test",
        data_output_path="../data/test_custom.csv",
        use_saved_tokenizer=True,
    )
