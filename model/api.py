import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from sklearn.feature_extraction.text import TfidfVectorizer
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import src.utils as utils
import pandas as pd
import re
import nltk
import numpy as np

nltk.download("stopwords")
nltk.download("punkt")
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import swifter
from sklearn.preprocessing import LabelEncoder

CONFIG_DATA = utils.config_load()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load your pre-trained ML model
class Model:
    def clean_data(self, data, CONFIG_DATA=CONFIG_DATA, return_file=True):
        # print("cleaning the data")

        # Lowering Case
        data[CONFIG_DATA["text_column"]] = data[CONFIG_DATA["text_column"]].str.lower()

        # Remove Non ASCII
        data[CONFIG_DATA["text_column"]] = (
            data[CONFIG_DATA["text_column"]]
            .str.encode("ascii", "ignore")
            .str.decode("ascii")
        )

        # Remove Whitespace in Start and End
        data[CONFIG_DATA["text_column"]] = data[CONFIG_DATA["text_column"]].str.strip()

        # Punctuation Removal Code
        data[CONFIG_DATA["text_column"]] = data[CONFIG_DATA["text_column"]].apply(
            lambda text: re.sub(
                r"[{}]".format(re.escape(string.punctuation)), "", str(text)
            )
        )

        # Remove Mention, Link, Hashtag, etc
        def replace_str(text):
            cleaned_text = re.sub(
                r"[\n\r\t]", " ", str(text)
            )  # Remove Tab, Enter, Space
            cleaned_text = re.sub(r"[^a-zA-Z0-9\s]", "", str(text))  # Remove Many Hwer
            cleaned_text = re.sub(r"\b[a-zA-Z]\b", "", str(text))  # Remove 1 Char Only
            cleaned_text = re.sub(r"\d+", "", str(text))  # Remove Number
            return cleaned_text

        data[CONFIG_DATA["text_column"]] = data[CONFIG_DATA["text_column"]].apply(
            replace_str
        )

        # Remove incomplete URL
        data[CONFIG_DATA["text_column"]] = (
            data[CONFIG_DATA["text_column"]]
            .replace("http://", " ")
            .replace("https://", " ")
        )

        # Remove Multiple Space
        data[CONFIG_DATA["text_column"]] = data[CONFIG_DATA["text_column"]].replace(
            r"\s+", " ", regex=True
        )

        # print("data was cleaned")
        if return_file:
            return data

    # Generate Preprocessor
    def generate_preprocessor(self, data, CONFIG_DATA=CONFIG_DATA, return_file=True):
        # clean data
        # print("ready to clean data")
        data = self.clean_data(data, CONFIG_DATA)

        # print("ready to preprocess")

        # tokenization
        def word_tokenize_wrapper(text):
            return word_tokenize(text)

        data[CONFIG_DATA["token_column"]] = data[CONFIG_DATA["text_column"]].apply(
            word_tokenize_wrapper
        )

        # Stopwords Removal (Filtering)
        list_stopwords = stopwords.words("indonesian")
        list_stopwords.extend(
            [
                "yg",
                "dg",
                "rt",
                "dgn",
                "ny",
                "d",
                "klo",
                "kalo",
                "amp",
                "biar",
                "bikin",
                "bilang",
                "gak",
                "ga",
                "krn",
                "nya",
                "nih",
                "sih",
                "si",
                "tau",
                "tdk",
                "tuh",
                "utk",
                "ya",
                "jd",
                "jgn",
                "sdh",
                "aja",
                "n",
                "t",
                "nyg",
                "hehe",
                "pen",
                "u",
                "nan",
                "loh",
                "rt",
                "&amp",
                "yah",
            ]
        )
        list_stopwords = set(list_stopwords)

        def stopwords_removal(words):
            return [word for word in words if word not in list_stopwords]

        # print("gagal stopwords")

        data[CONFIG_DATA["token_column"]] = data[CONFIG_DATA["token_column"]].apply(
            stopwords_removal
        )
        # Stemming
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()

        def stemmed_wrapper(term):
            return stemmer.stem(term)

        term_dict = {}

        for document in data[CONFIG_DATA["token_column"]]:
            for term in document:
                if term not in term_dict:
                    term_dict[term] = " "

        for term in term_dict:
            term_dict[term] = stemmed_wrapper(term)

        def get_stemmed_term(document):
            return [term_dict[term] for term in document]

        # print("gagal stemming")

        data[CONFIG_DATA["token_column"]] = data[
            CONFIG_DATA["token_column"]
        ].swifter.apply(get_stemmed_term)

        # print('data was processed')
        if return_file:
            return data

    def predict(self, X):
        """Function to predict the data"""
        # Preprocess data
        # print("ready to go")
        X_clean = self.generate_preprocessor(X)  # Use generate_preprocessor
        # Convert tokenized texts to string format
        # print("ready to run model")
        X_clean = [" ".join(tokens) for tokens in X_clean[CONFIG_DATA["token_column"]]]
        # Create TF-IDF vectorizer
        tfidf_vectorizer = utils.pickle_load(CONFIG_DATA["tfidf_vectorizer_path"])
        # Transform the data into TF-IDF features
        # print("ready to tfidf")
        X_clean = tfidf_vectorizer.transform(X_clean)
        # Predict data
        # print("ready to predict")
        model = utils.pickle_load(CONFIG_DATA["best_model_path"])
        y_pred = model.predict_proba(X_clean)
        y_pred.tolist()

        # Create Label Encoder
        label_encoder = utils.pickle_load(CONFIG_DATA["label_encoder_path"])

        # Transform class indices to original class labels
        class_labels = label_encoder.inverse_transform(
            np.arange(len(label_encoder.classes_))
        )

        # Create a list to store class labels with probabilities
        class_prob_list = []

        # Iterate through each input sample and append class label with probability to the list
        for row in y_pred:
            class_probabilities = [
                {"label": label, "probability": float(prob)}
                for label, prob in zip(class_labels, row)
            ]
            class_prob_list.append(class_probabilities)

        # Return the list of dictionaries containing class labels and probabilities
        return class_prob_list


model = Model()


# Testing Index
@app.get("/", response_class=HTMLResponse)
def read_root():
    template_path = os.path.join(
        os.path.dirname(__file__), "src", "templates", "index.html"
    )
    return open(template_path).read()


# Get Predict
@app.post("/predict")
def predict_text(data: dict):
    try:
        text = data.get("text")
        if not text:
            raise HTTPException(
                status_code=400, detail="Missing 'text' field in request payload"
            )

        # Preprocess data
        input_data = pd.DataFrame({"content": [text]})
        predictions = model.predict(input_data)  # Use the model to make predictions

        # Return predictions for each class
        return {"class_probabilities": predictions}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
