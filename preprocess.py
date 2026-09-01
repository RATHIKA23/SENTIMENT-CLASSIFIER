# preprocess.py

import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('stopwords', quiet=True)
STOP_WORDS = set(stopwords.words('english'))


def remove_stopwords(text):
    """Strip stopwords — use for LSTM path only, not BERT."""
    return ' '.join([word for word in text.split() if word not in STOP_WORDS])


def tokenize_and_pad(train_texts, test_texts, vocab_size=10000, max_len=200):
    """Fit tokenizer on train texts only, then pad/truncate both sets."""
    tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
    tokenizer.fit_on_texts(train_texts)

    train_seq = tokenizer.texts_to_sequences(train_texts)
    test_seq = tokenizer.texts_to_sequences(test_texts)

    train_pad = pad_sequences(train_seq, maxlen=max_len, padding='post', truncating='post')
    test_pad = pad_sequences(test_seq, maxlen=max_len, padding='post', truncating='post')

    return train_pad, test_pad, tokenizer
