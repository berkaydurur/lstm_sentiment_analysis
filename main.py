import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Embedding,LSTM
import tensorflow as tf
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.models import load_model
from sklearn.model_selection import train_test_split
from keras import backend as K
import re
import nltk
import glob



stop_word_list=['acaba', 'ama', 'aslında', 'az', 'bazı', 'belki', 'biri', 'birkaç', 'birşey', 'biz', 'bu', 'çok', 'çünkü', 'da', 'daha', 'de', 'defa', 'diye', 'eğer', 'en', 'gibi', 'hem', 'hep', 'hepsi', 'her', 'hiç', 'için', 'ile', 'ise', 'kez', 'ki', 'kim', 'mı', 'mu', 'mü', 'nasıl', 'ne', 'neden', 'nerde', 'nerede', 'nereye', 'niçin', 'niye', 'o', 'sanki', 'şey', 'siz', 'şu', 'tüm', 've', 'veya', 'ya', 'yani']
pathArr=glob.glob(r"C:\TweetData\ruh_hali\raw_texts\uzgun\*.txt")
data = pd.DataFrame(columns=[0, 'Sentiment'])

for i in pathArr:
    df = pd.read_csv(i,delimiter='\t', header = None,encoding="windows-1254")
    df['Sentiment'] = int(0)
    df[0] = df[0].apply(lambda x: x.lower())
    df[0] = df[0].apply(lambda x: re.sub('[,\.!?:()"]', '', x))
    df[0] = df[0].apply(lambda x: x.strip())

    def token(values):
        words = nltk.tokenize.word_tokenize(values)
        filtered_words = [word for word in words if word not in stop_word_list]
        not_stopword_doc = " ".join(filtered_words)
        return not_stopword_doc


    df[0] = df[0].apply(lambda x: token(x))
    data = data.append(df, ignore_index=True)

pathArr=glob.glob(r"C:\TweetData\ruh_hali\raw_texts\neseli\*.txt")
for i in pathArr:
    df = pd.read_csv(i,delimiter='\t', header = None,encoding="windows-1254")
    df['Sentiment'] = int(1)
    df[0] = df[0].apply(lambda x: x.lower())
    df[0] = df[0].apply(lambda x: re.sub('[,\.!?:()"]', '', x))
    df[0] = df[0].apply(lambda x: x.strip())

    def token(values):
        words = nltk.tokenize.word_tokenize(values)
        filtered_words = [word for word in words if word not in stop_word_list]
        not_stopword_doc = " ".join(filtered_words)
        return not_stopword_doc


    df[0] = df[0].apply(lambda x: token(x))
    data=data.append(df,sort=False)
pathArr=glob.glob(r"C:\TweetData\ruh_hali\raw_texts\sinirli\*.txt")
for i in pathArr:
    df = pd.read_csv(i,delimiter='\t', header = None,encoding="windows-1254")
    df['Sentiment'] = int(2)
    df[0] = df[0].apply(lambda x: x.lower())
    df[0] = df[0].apply(lambda x: re.sub('[,\.!?:()"]', '', x))
    df[0] = df[0].apply(lambda x: x.strip())

    def token(values):
        words = nltk.tokenize.word_tokenize(values)
        filtered_words = [word for word in words if word not in stop_word_list]
        not_stopword_doc = " ".join(filtered_words)
        return not_stopword_doc


    df[0] = df[0].apply(lambda x: token(x))
    data = data.append(df, ignore_index=True)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(data)

dataVal = data[0].values.tolist()
sentimentVal = data["Sentiment"].values.tolist()
x_train, x_test, y_train, y_test = train_test_split(dataVal, sentimentVal, test_size=0.6
                                                    , random_state=42)

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(dataVal)
tokenizer.word_index

x_train_tokens = tokenizer.texts_to_sequences(x_train)
x_test_tokens = tokenizer.texts_to_sequences(x_test)


num_tokens = [len(tokens) for tokens in x_train_tokens + x_test_tokens]
num_tokens = np.array(num_tokens)

max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
max_tokens = int(max_tokens)


np.sum(num_tokens < max_tokens) / len(num_tokens)

x_train_pad = pad_sequences(x_train_tokens, maxlen=max_tokens)
x_test_pad = pad_sequences(x_test_tokens, maxlen=max_tokens)


idx = tokenizer.word_index
inverse_map = dict(zip(idx.values(), idx.keys()))


def tokens_to_string(tokens):
    words = [inverse_map[token] for token in tokens if token!=0]
    text = ' '.join(words)
    return text

model = Sequential()

embedding_size = 50

model.add(Embedding(input_dim=10000,
                    output_dim=embedding_size,
                    input_length=max_tokens,
                    name='embedding_layer'))

model.add(LSTM(units=16, return_sequences=True))

model.add(LSTM(units=8, return_sequences=True))
model.add(LSTM(units=4))
model.add(Dense(1, activation='sigmoid'))
optimizer = tf.keras.optimizers.Adam(lr=1e-3)
model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

#model.summary()
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc',f1_m,precision_m, recall_m])

model.fit(np.array(x_train_pad), np.array(y_train),validation_split=0.2, epochs=25, batch_size=64)


loss, accuracy, f1_score, precision, recall = model.evaluate(np.array(x_train_pad), np.array(y_train))


print(loss,f1_score)

