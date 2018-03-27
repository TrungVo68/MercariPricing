import time
import numpy as np
import pandas as pd
import math

from nltk.corpus import stopwords

from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dropout, Dense, concatenate, GRU, Embedding, Flatten, Activation, LSTM
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K


# ==========================================================================
# ALL FUNCTIONS HERE
# ==========================================================================
# CALCULATING ROOT MEAN SQUARE LOG ERROR
def rmsle1(y, y_pred):
    assert len(y) == len(y_pred)
    terms_to_sum  = [(math.log(y_pred[i]+1) - math.log(y[i]+1))**2 for i, pred in enumerate(y_pred)]
    return(sum(terms_to_sum) * (1/len(y)))**0.5

# Y and Y_pred will already be in log scale by the time this is used, so no need to log them in the function
def rmsle2(Y, Y_pred):
    assert Y.shape == Y_pred.shape
    return np.sqrt(np.mean(np.square(Y_pred - Y )))
    
# COUNT HOW MANY WORDS IN A SENTENCE
def word_count(text):
    try:
        if (text == "No description yet") | (text == "missing"):
            return 0
        else:
            text = text.lower()
            words = [w for w in text.split(" ")]
            return len(words)
    except:
        return 0

# SPLIT CATEGORY_NAME INTO 3 PARTS
def split_cat_name(text):
    try: 
        return text.split("/")
    except: 
        return ("No Label", "No Label", "No Label")

# TRYING TO FIND AND IMPUTE BRAND NAME
def brandfinder(line):
    brand = line[0]
    name = line[1]
    namesplit = name.split(' ')
    if brand == 'missing':
        for x in namesplit:
            if x in all_brands:
                return name
    if name in all_brands:
        return name
    return brand

# Filling missing values
def fill_missing_values(df):
    df.category_name.fillna(value="missing", inplace=True)
    df.brand_name.fillna(value="missing", inplace=True)
    df.item_description.fillna(value="missing", inplace=True)
    df.item_description.replace('No description yet',"missing", inplace=True)
    return df

# GET DATA FOR RNN MODEL
def prepare_rnn_data(dataset):
    X = {
        'name': pad_sequences(dataset.seq_name, maxlen=MAX_NAME_SEQ),
        'item_desc': pad_sequences(dataset.seq_item_desc, maxlen=MAX_ITEM_DESC_SEQ),
        'brand_name': np.array(dataset.brand_name),
        'category': np.array(dataset.category),
#         'category_name': pad_sequences(dataset.seq_category, maxlen=MAX_CATEGORY_SEQ),
        'item_condition': np.array(dataset.item_condition_id),
        'num_vars': np.array(dataset[["shipping"]]),
        'desc_len': np.array(dataset[["description_length"]]),
        'name_len': np.array(dataset[["name_length"]]),
        'subcat_0': np.array(dataset.subcat0),
        'subcat_1': np.array(dataset.subcat1),
        'subcat_2': np.array(dataset.subcat2),
    }
    return X


# ==========================================================================
# LOAD FILE HERE
# ==========================================================================    
path = "D:/Kaggle/Python/MercariPricing/"
#path = "../input/"
train = pd.read_csv(f'{path}train.tsv', sep='\t')
test = pd.read_csv(f'{path}test.tsv', sep='\t') 

# ==========================================================================
# DATA CLEANSING AND FEATURE ENGINEERING
# ==========================================================================    
# REMOVE ALL ROWS THAT HAVE LESS THAN $3 IN PRICE
train[train.price<3].shape
train = train.drop(train[train.price < 3].index)

# IMPUTE NAN VALUE 
train.isnull().sum()
train.brand_name.fillna(value="missing", inplace=True)
test.brand_name.fillna(value="missin", inplace=True)

# REMOVING STOPWORDS
stop_word = stopwords.words("english")
#train.item_description.fillna(value="No description yet", inplace=True)
train['item_description'] = train['item_description'].apply(lambda x: ' '.join([w for w in x.split() if w not in stop_word]))
train.name = train.name.apply(lambda x: ' '.join([w for w in x.split() if w not in stop_word]))

train.name.isnull().sum()
test['item_description'] = test['item_description'].apply(lambda x: ' '.join([w for w in x.split() if w not in stop_word]))
test['name'] = test['name'].apply(lambda x: ' '.join([w for w in x.split() if w not in stop_word]))

# ADDING 2 FEATURES: ITEM_DESCRIPTION LENGTH AND NAME_LENGTH
train['name_length'] = train.name.apply(lambda x: word_count(x))
test['name_length'] = test.name.apply(lambda x: word_count(x))
train['description_length'] = train.item_description.apply(lambda x: word_count(x))
test['description_length'] = test.item_description.apply(lambda x: word_count(x))

# CREATING 3 NEW FEATURE COLUMN FROM CATEGORY_NAME
train['subcat0'], train['subcat1'], train['subcat2'] = zip(*train['category_name'].apply(lambda x: split_cat_name(x)))
test['subcat0'], test['subcat1'], test['subcat2'] = zip(*test.category_name.apply(lambda x: split_cat_name(x)))

# IMPUTE MISSING BRAND_NAME
full_set = pd.concat([train,test])
all_brands = set(full_set.brand_name.values)

train['brand_name'] = train[['brand_name', 'name']].apply(brandfinder, axis = 1)
test['brand_name'] = test[['brand_name', 'name']].apply(brandfinder, axis = 1)

# TAKE LOG FOR PRICE TO REDUCE SKEWNESS
train['target'] = np.log1p(train.price)

# SPLIT TRAIN DATA INTO TRAINING 
df_train, df_eval = train_test_split(train, random_state=68, train_size=0.8)

n_train = df_train.shape[0]
n_eval = df_eval.shape[0]
n_test = test.shape[0]

# COMBINING TRAIN, EVAL AND TEST SET TOGETHER
df = pd.concat([df_train, df_eval, test])
df = fill_missing_values(df)
df.head()

# PROCESS CATEGORICAL PARAMETERS
le = LabelEncoder()
#le.fit(df.category_name)
#df['category'] = le.transform(df.category_name)
df['category'] = le.fit_transform(df.category_name)
df.brand_name = le.fit_transform(df.brand_name)
df.subcat0 = le.fit_transform(df.subcat0)
df.subcat1 = le.fit_transform(df.subcat1)
df.subcat2 = le.fit_transform(df.subcat2)

# TRANSFORMING TEXT DATA TO SEQUENCE
raw_text = np.hstack([df.item_description.str.lower(), df.name.str.lower(), df.category_name.str.lower()])

token_raw = Tokenizer()
token_raw.fit_on_texts(raw_text)

df['seq_item_desc'] = token_raw.texts_to_sequences(df.item_description.str.lower())
df['seq_name'] = token_raw.texts_to_sequences(df.name.str.lower())
del token_raw


# ==========================================================================
# PREPARING FOR RNN MODEL
# ========================================================================== 
MAX_NAME_SEQ = 10 #17
MAX_ITEM_DESC_SEQ = 75 #269
MAX_CATEGORY_SEQ = 8 #8
MAX_TEXT = np.max([np.max(df.seq_name.max()), np.max(df.seq_item_desc.max())]) + 100
#     np.max(full_df.seq_category.max()),
#]) + 100
MAX_CATEGORY = np.max(df.category.max()) + 1
MAX_BRAND = np.max(df.brand_name.max()) + 1
MAX_CONDITION = np.max(df.item_condition_id.max()) + 1
MAX_DESC_LEN = np.max(df.description_length.max()) + 1
MAX_NAME_LEN = np.max(df.name_length.max()) + 1
MAX_SUBCAT_0 = np.max(df.subcat0.max()) + 1
MAX_SUBCAT_1 = np.max(df.subcat1.max()) + 1
MAX_SUBCAT_2 = np.max(df.subcat2.max()) + 1

# SPLIT DATA INTO TRAIN, EVAL AND TEST SET AS INPUT TO RNN MODEL
df_train = df[:n_train]
df_eval = df[n_train:n_train+n_eval]
df_test = df[n_train+n_eval:]

X_train = prepare_rnn_data(df_train)
Y_train = df_train.target.values.reshape(-1,1)

X_eval = prepare_rnn_data(df_eval)
Y_eval = df_eval.target.values.reshape(-1,1)

X_test = prepare_rnn_data(df_test)


# ==========================================================================
# RUN RNN MODEL
# ========================================================================== 
# Set hyper parameters for the model.
BATCH_SIZE = 512 * 3
epochs = 2

# Calculate learning rate decay.
exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
steps = int(len(X_train['name']) / BATCH_SIZE) * epochs
lr_initial, lr_final = 0.005, 0.001
lr_decay = exp_decay(lr_initial, lr_final, steps)



# RNN MODEL
def RNN_nodel(lr=0.001, decay=0.0):
    # DEFINE THE LEGTH OF EACH INPUT COLUMN
    name = Input(shape=[X_train["name"].shape[1]], name="name")
    item_desc = Input(shape=[X_train["item_desc"].shape[1]], name="item_desc")
    brand_name = Input(shape=[1], name="brand_name")
#     category = Input(shape=[1], name="category")
#     category_name = Input(shape=[X_train["category_name"].shape[1]], name="category_name")
    item_condition = Input(shape=[1], name="item_condition")
    num_vars = Input(shape=[X_train["num_vars"].shape[1]], name="num_vars")
    desc_len = Input(shape=[1], name="desc_len")
    name_len = Input(shape=[1], name="name_len")
    subcat_0 = Input(shape=[1], name="subcat_0")
    subcat_1 = Input(shape=[1], name="subcat_1")
    subcat_2 = Input(shape=[1], name="subcat_2")

    # CONVERTING WORDS INTO VECTOR VIA EMBEDDING LAYERS
    emb_name = Embedding(MAX_TEXT, 20)(name)
    emb_item_desc = Embedding(MAX_TEXT, 60)(item_desc)
    emb_brand_name = Embedding(MAX_BRAND, 10)(brand_name)
#     emb_category_name = Embedding(MAX_TEXT, 20)(category_name)
#     emb_category = Embedding(MAX_CATEGORY, 10)(category)
    emb_item_condition = Embedding(MAX_CONDITION, 5)(item_condition)
    emb_desc_len = Embedding(MAX_DESC_LEN, 5)(desc_len)
    emb_name_len = Embedding(MAX_NAME_LEN, 5)(name_len)
    emb_subcat_0 = Embedding(MAX_SUBCAT_0, 10)(subcat_0)
    emb_subcat_1 = Embedding(MAX_SUBCAT_1, 10)(subcat_1)
    emb_subcat_2 = Embedding(MAX_SUBCAT_2, 10)(subcat_2)
    
    # RNN LAYER, CAN USE GRU OR LSTM OR ....
    rnn_layer1 = GRU(16) (emb_item_desc)
    rnn_layer2 = GRU(8) (emb_name)
    
#    rnn_layer1 = LSTM(32) (emb_item_desc)
#    rnn_layer2 = LSTM(16) (emb_name)
    
#     rnn_layer3 = GRU(8) (emb_category_name)

#    MAIN LAYER, FLATTEN OUT ALL VECTORS BEFORE SENDING THROUGH TO FULLY
#    CONNECTED LAYER OF A NEURAL NETWORK
    main_l = concatenate([
        Flatten() (emb_brand_name)
#         , Flatten() (emb_category)
        , Flatten() (emb_item_condition)
        , Flatten() (emb_desc_len)
        , Flatten() (emb_name_len)
        , Flatten() (emb_subcat_0)
        , Flatten() (emb_subcat_1)
        , Flatten() (emb_subcat_2)
        , rnn_layer1
        , rnn_layer2
#         , rnn_layer3
        , num_vars
    ])
    
#    SETTING UP DROP-OUT RATE AND FULLY CONNECTED LAYER (DENSE)
    main_l = Dropout(0.1)(Dense(512,kernel_initializer='normal',activation='relu') (main_l))
#    main_l = Dropout(0.1)(Dense(256,kernel_initializer='normal',activation='relu') (main_l))
    main_l = Dropout(0.1)(Dense(128,kernel_initializer='normal',activation='relu') (main_l))
#    main_l = Dropout(0.1)(Dense(64,kernel_initializer='normal',activation='relu') (main_l))

#    OUTPUT LAYER COMPRISES EMBEDDING (CONVERT WORDS TO VEC, RNN, FLATTEN, DROP-OUT AND DENSE LAYERS)
    output = Dense(1, activation="linear") (main_l)
    
#    SETUP RNN MODEL
    model = Model([name, item_desc, brand_name , item_condition, 
                   num_vars, desc_len, name_len, subcat_0, subcat_1, subcat_2], output)
    
#    SETUP OPTIMIER FOR A RNN MODEL USING ADAM USING ADAM & PRE-DETERMINED LEARNING RATE
    optimizer = Adam(lr=lr, decay=decay)  
    model.compile(loss = 'mse', optimizer = optimizer)

    return(model)
    
#model = RNN_nodel()
#model.summary()
#del model
    
    
# Create model and fit it with training dataset.
rnn_model = RNN_nodel(lr=lr_initial, decay=lr_decay)
rnn_model.fit(X_train, Y_train, epochs=epochs, batch_size=BATCH_SIZE, validation_data=(X_eval, Y_eval), verbose=1)

# Evaluate RNN model on dev data
print("Evaluating the model on validation data...")
Y_eval_preds_rnn = rnn_model.predict(X_eval, batch_size=BATCH_SIZE)
print(" RMSLE error:", rmsle2(Y_eval, Y_eval_preds_rnn))

# Make prediction for test data
rnn_preds = rnn_model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
rnn_preds = np.expm1(rnn_preds)
























