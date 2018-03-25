# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 23:03:40 2018

@author: trung
"""
import time
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dropout, Dense, concatenate, GRU, Embedding, Flatten, Activation
# from keras.layers import Bidirectional
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
from nltk.corpus import stopwords
import math

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
        if text == "No description yet":
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
def get_rnn_data(dataset):
    X = {
        'name': pad_sequences(dataset.seq_name, maxlen=MAX_NAME_SEQ),
        'item_desc': pad_sequences(dataset.seq_item_description, maxlen=MAX_ITEM_DESC_SEQ),
        'brand_name': np.array(dataset.brand_name),
        'category': np.array(dataset.category),
#         'category_name': pad_sequences(dataset.seq_category, maxlen=MAX_CATEGORY_SEQ),
        'item_condition': np.array(dataset.item_condition_id),
        'num_vars': np.array(dataset[["shipping"]]),
        'desc_len': np.array(dataset[["desc_len"]]),
        'name_len': np.array(dataset[["name_len"]]),
        'subcat_0': np.array(dataset.subcat_0),
        'subcat_1': np.array(dataset.subcat_1),
        'subcat_2': np.array(dataset.subcat_2),
    }
    return X

#Here are some unused RMSE and RMSLE functions. They can be used as a loss function in the model 
#but the built in 'mse' works just as well. Worth having around just in case. 
#There needs to be a small non-zero number added to the means or a zero value might sneak in and 
#cause it to return NaN. As is, they work fine as a loss function just not special.
def root_mean_squared_logarithmic_error(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
    return K.sqrt(K.mean(K.square(first_log - second_log), axis=-1)+0.0000001)
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)+0.0000001)

# RNN MODEL
def new_rnn_model(lr=0.001, decay=0.0):
    # Inputs
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

    # Embeddings layers (adjust outputs to help model)
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
    
    # rnn layers (GRUs are faster than LSTMs and speed is important here)
    rnn_layer1 = GRU(16) (emb_item_desc)
    rnn_layer2 = GRU(8) (emb_name)
#     rnn_layer3 = GRU(8) (emb_category_name)

    # main layers
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
    # (incressing the nodes or adding layers does not effect the time quite as much as the rnn layers)
    main_l = Dropout(0.1)(Dense(512,kernel_initializer='normal',activation='relu') (main_l))
#    main_l = Dropout(0.1)(Dense(256,kernel_initializer='normal',activation='relu') (main_l))
    main_l = Dropout(0.1)(Dense(128,kernel_initializer='normal',activation='relu') (main_l))
#    main_l = Dropout(0.1)(Dense(64,kernel_initializer='normal',activation='relu') (main_l))

    # the output layer.
    output = Dense(1, activation="linear") (main_l)
    
    model = Model([name, item_desc, brand_name , item_condition, 
                   num_vars, desc_len, name_len, subcat_0, subcat_1, subcat_2], output)

    optimizer = Adam(lr=lr, decay=decay)
    # (mean squared error loss function works as well as custom functions)  
    model.compile(loss = 'mse', optimizer = optimizer)

    return(model)

# RIDGE PREPROCESSOR
def build_preprocessor(field):
    field_idx = list(full_df.columns).index(field)
    return lambda x: default_preprocessor(x[field_idx])

# CHOOSE THE WEIGHT OF THE 3 MODELS
def aggregate_predicts3(Y1, Y2, Y3, ratio1, ratio2):
    assert Y1.shape == Y2.shape
    return(Y1 * ratio1 + Y2 * ratio2 + Y3 * (1.0 - ratio1-ratio2))

def aggregate_predicts2(Y1, Y2,ratio):
    assert Y1.shape == Y2.shape
    return Y1 * ratio + Y2 * (1.0 - ratio)
# ==========================================================================
# LOAD FILE HERE
# ==========================================================================    
path = "D:/Kaggle/Python/MercariPricing/"
#path = "../input/"
train_df = pd.read_csv(f'{path}train.tsv', sep='\t')
test_df = pd.read_csv(f'{path}test.tsv', sep='\t') 
    
    
# ==========================================================================
# DATA CLEANSING
# ==========================================================================    
# MERCARI DOES NOT ALLOW ANYTHING LESS THAN $3 AND VOER $2000
train_df = train_df.drop(train_df[(train_df.price < 3.0)].index)

# REMOVE STOP WORDS    
stop = stopwords.words('english')
train_df.item_description.fillna(value='No description yet', inplace=True)
train_df['item_description'] = train_df['item_description'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
train_df.name.fillna(value="missing", inplace=True)
train_df['name'] = train_df['name'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

test_df.item_description.fillna(value='No description yet', inplace=True)
test_df['item_description'] = test_df['item_description'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
test_df.name.fillna(value="missing", inplace=True)
test_df['name'] = test_df['name'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    

# CREATE ADDITIONAL FEATURES USING THE LENGTH OF NAME AND ITEM DESCRIPTION   
train_df['desc_len'] = train_df['item_description'].apply(lambda x: word_count(x))
test_df['desc_len'] = test_df['item_description'].apply(lambda x: word_count(x))
train_df['name_len'] = train_df['name'].apply(lambda x: word_count(x))
test_df['name_len'] = test_df['name'].apply(lambda x: word_count(x))


# SPLIT CATEGORY_NAME INTO 3 SECTIONS
train_df['subcat_0'], train_df['subcat_1'], train_df['subcat_2'] = zip(*train_df['category_name'].apply(lambda x: split_cat_name(x)))
test_df['subcat_0'], test_df['subcat_1'], test_df['subcat_2'] = zip(*test_df['category_name'].apply(lambda x: split_cat_name(x)))


# IMPUTE MISSING TO BRAND_NAME    
full_set = pd.concat([train_df,test_df])
all_brands = set(full_set['brand_name'].values)
train_df.brand_name.fillna(value="missing", inplace=True)
test_df.brand_name.fillna(value="missing", inplace=True)
   
#premissing = len(train_df.loc[train_df['brand_name'] == 'missing'])  
train_df['brand_name'] = train_df[['brand_name','name']].apply(brandfinder, axis = 1)
test_df['brand_name'] = test_df[['brand_name','name']].apply(brandfinder, axis = 1)
#found = premissing-len(train_df.loc[train_df['brand_name'] == 'missing'])    
    
    
# ==========================================================================
# SPLIT DATA FOR TRAINING, TESTING AND VALIDATING
# ==========================================================================  
# Scale target variable to log.
train_df["target"] = np.log1p(train_df.price)

# Split training examples into train/dev examples.
train_df, dev_df = train_test_split(train_df, random_state=68, train_size=0.90)

# Calculate number of train/dev/test examples.
n_trains = train_df.shape[0]
n_devs = dev_df.shape[0]
n_tests = test_df.shape[0]

# Concatenate train - dev - test data for easy to handle
full_df = pd.concat([train_df, dev_df, test_df])  

# FILLING MISSING VALUES 
full_df = fill_missing_values(full_df)

# PROCESSING CATEGORICAL VARIABLE
le = LabelEncoder()
# full_df.category = full_df.category_name
le.fit(full_df.category_name)
full_df['category'] = le.transform(full_df.category_name)

le.fit(full_df.brand_name)
full_df.brand_name = le.transform(full_df.brand_name)

le.fit(full_df.subcat_0)
full_df.subcat_0 = le.transform(full_df.subcat_0)

le.fit(full_df.subcat_1)
full_df.subcat_1 = le.transform(full_df.subcat_1)

le.fit(full_df.subcat_2)
full_df.subcat_2 = le.transform(full_df.subcat_2)


# TRANSFORMING TEXT DATA TO SEQUENCE
print("Transforming text data to sequences...")
raw_text = np.hstack([full_df.item_description.str.lower(), full_df.name.str.lower(), full_df.category_name.str.lower()])

print("   Fitting tokenizer...")
tok_raw = Tokenizer()
tok_raw.fit_on_texts(raw_text)

print("   Transforming text to sequences...")
full_df['seq_item_description'] = tok_raw.texts_to_sequences(full_df.item_description.str.lower())
full_df['seq_name'] = tok_raw.texts_to_sequences(full_df.name.str.lower())
# full_df['seq_category'] = tok_raw.texts_to_sequences(full_df.category_name.str.lower())
del tok_raw


# ==========================================================================
# RNN
# ========================================================================== 
MAX_NAME_SEQ = 10 #17
MAX_ITEM_DESC_SEQ = 75 #269
MAX_CATEGORY_SEQ = 8 #8
MAX_TEXT = np.max([
    np.max(full_df.seq_name.max()),
    np.max(full_df.seq_item_description.max())
#     np.max(full_df.seq_category.max()),
]) + 100
MAX_CATEGORY = np.max(full_df.category.max()) + 1
MAX_BRAND = np.max(full_df.brand_name.max()) + 1
MAX_CONDITION = np.max(full_df.item_condition_id.max()) + 1
MAX_DESC_LEN = np.max(full_df.desc_len.max()) + 1
MAX_NAME_LEN = np.max(full_df.name_len.max()) + 1
MAX_SUBCAT_0 = np.max(full_df.subcat_0.max()) + 1
MAX_SUBCAT_1 = np.max(full_df.subcat_1.max()) + 1
MAX_SUBCAT_2 = np.max(full_df.subcat_2.max()) + 1


# PREPARE DATA FOR RNN
train = full_df[:n_trains]
dev = full_df[n_trains:n_trains+n_devs]
test = full_df[n_trains+n_devs:]

X_train = get_rnn_data(train)
Y_train = train.target.values.reshape(-1, 1)

X_dev = get_rnn_data(dev)
Y_dev = dev.target.values.reshape(-1, 1)

X_test = get_rnn_data(test)


# RNN MODEL
np.random.seed(68)
#model = new_rnn_model()
#model.summary()
#del model

# Set hyper parameters for the model.
BATCH_SIZE = 512 * 3
epochs = 2

# Calculate learning rate decay.
exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
steps = int(len(X_train['name']) / BATCH_SIZE) * epochs
lr_init, lr_fin = 0.005, 0.001
lr_decay = exp_decay(lr_init, lr_fin, steps)

# Create model and fit it with training dataset.
rnn_model = new_rnn_model(lr=lr_init, decay=lr_decay)
rnn_model.fit(
        X_train, Y_train, epochs=epochs, batch_size=BATCH_SIZE,
        validation_data=(X_dev, Y_dev), verbose=1,
)

# Evaluate RNN model on dev data
print("Evaluating the model on validation data...")
Y_dev_preds_rnn = rnn_model.predict(X_dev, batch_size=BATCH_SIZE)
print(" RMSLE error:", rmsle2(Y_dev, Y_dev_preds_rnn))

# Make prediction for test data
rnn_preds = rnn_model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
rnn_preds = np.expm1(rnn_preds)


# ==========================================================================
# RIDGE
# ==========================================================================
full_df = pd.concat([train_df, dev_df, test_df])

print("Handling missing values...")
full_df['category_name'] = full_df['category_name'].fillna('missing').astype(str)
full_df['subcat_0'] = full_df['subcat_0'].astype(str)
full_df['subcat_1'] = full_df['subcat_1'].astype(str)
full_df['subcat_2'] = full_df['subcat_2'].astype(str)
full_df['brand_name'] = full_df['brand_name'].fillna('missing').astype(str)
full_df['shipping'] = full_df['shipping'].astype(str)
full_df['item_condition_id'] = full_df['item_condition_id'].astype(str)
full_df['desc_len'] = full_df['desc_len'].astype(str)
full_df['name_len'] = full_df['name_len'].astype(str)
full_df['item_description'] = full_df['item_description'].fillna('No description yet').astype(str)

print("Vectorizing data...")
default_preprocessor = CountVectorizer().build_preprocessor()
#default_preprocessor = HashingVectorizer().build_preprocessor()

vectorizer = FeatureUnion([
    ('name', CountVectorizer(
        ngram_range=(1, 2),
        max_features=50000,
        preprocessor=build_preprocessor('name'))),
#     ('category_name', CountVectorizer(
#         token_pattern='.+',
#         preprocessor=build_preprocessor('category_name'))),
    ('subcat_0', CountVectorizer(
        token_pattern='.+',
        preprocessor=build_preprocessor('subcat_0'))),
    ('subcat_1', CountVectorizer(
        token_pattern='.+',
        preprocessor=build_preprocessor('subcat_1'))),
    ('subcat_2', CountVectorizer(
        token_pattern='.+',
        preprocessor=build_preprocessor('subcat_2'))),
    ('brand_name', CountVectorizer(
        token_pattern='.+',
        preprocessor=build_preprocessor('brand_name'))),
    ('shipping', CountVectorizer(
        token_pattern='\d+',
        preprocessor=build_preprocessor('shipping'))),
    ('item_condition_id', CountVectorizer(
        token_pattern='\d+',
        preprocessor=build_preprocessor('item_condition_id'))),
    ('desc_len', CountVectorizer(
        token_pattern='\d+',
        preprocessor=build_preprocessor('desc_len'))),
    ('name_len', CountVectorizer(
        token_pattern='\d+',
        preprocessor=build_preprocessor('name_len'))),
    ('item_description', TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=100000,
        preprocessor=build_preprocessor('item_description'))),
])

# CREATING TRAIN, DEV AND TEST DATA SET
X = vectorizer.fit_transform(full_df.values)

X_train = X[:n_trains]
Y_train = train_df.target.values.reshape(-1, 1)

X_dev = X[n_trains:n_trains+n_devs]
Y_dev = dev_df.target.values.reshape(-1, 1)

X_test = X[n_trains+n_devs:]
print(X.shape, X_train.shape, X_dev.shape, X_test.shape)

# TRAINING RIDGE MODEL
print("Fitting Ridge model on training examples...")

#ridge_model = Ridge(
#    solver='auto', fit_intercept=True, alpha=1.0,
#    max_iter=100, normalize=False, tol=0.05, random_state = 1,
#)

ridge_modelCV = RidgeCV(
    fit_intercept=True, alphas=[5.0],
    normalize=False, cv = 2, scoring='neg_mean_squared_error',
)

#start = time.time()
#ridge_model.fit(X_train, Y_train)
#print("ridge training time: "+str(time.time()-start))

ridge_modelCV.fit(X_train, Y_train)

# EVALUATING RIDGE AND RIDGECV MODELS
#Y_dev_preds_ridge = ridge_model.predict(X_dev)
#Y_dev_preds_ridge = Y_dev_preds_ridge.reshape(-1, 1)
#print("RMSLE on dev set:", rmsle2(Y_dev, Y_dev_preds_ridge))

Y_dev_preds_ridgeCV = ridge_modelCV.predict(X_dev)
Y_dev_preds_ridgeCV = Y_dev_preds_ridgeCV.reshape(-1, 1)
print("CV RMSLE on dev set:", rmsle2(Y_dev, Y_dev_preds_ridgeCV))

# RIDGE AND RIDGECV PREDICTION
#ridge_preds = ridge_model.predict(X_test)
#ridge_preds = np.expm1(ridge_preds)
ridgeCV_preds = ridge_modelCV.predict(X_test)
ridgeCV_preds = np.expm1(ridgeCV_preds)

# FIND THE OPTIMAL WEIGHT BETWEEN THE 3 MODELS
#ratio optimum finder for 3 models
#best1 = 0
#best2 = 0
#lowest = 0.99
#for i in range(100):
#    for j in range(100):
#        r = i*0.01
#        r2 = j*0.01
#        if r+r2 < 1.0:
#            Y_dev_preds = aggregate_predicts3(Y_dev_preds_rnn, Y_dev_preds_ridgeCV, Y_dev_preds_ridge, r, r2)
#            fpred = rmsle2(Y_dev, Y_dev_preds)
#            if fpred < lowest:
#                best1 = r
#                best2 = r2
#                lowest = fpred
#             print(str(r)+"-RMSL error for RNN + Ridge + RidgeCV on dev set:", fpred)
#Y_dev_preds = aggregate_predicts3(Y_dev_preds_rnn, Y_dev_preds_ridgeCV, Y_dev_preds_ridge, best1, best2)
 
#ratio optimum finder for 2 models
best = 0
lowest = 0.99
for i in range(100):
    r = i*0.01
    Y_dev_preds = aggregate_predicts2(Y_dev_preds_rnn, Y_dev_preds_ridgeCV, r)
    fpred = rmsle2(Y_dev, Y_dev_preds)
    if fpred < lowest:
        best = r
        lowest = fpred
#    print(str(r)+"-RMSL error for RNN + Ridge on dev set:", fpred)
Y_dev_preds = aggregate_predicts2(Y_dev_preds_rnn, Y_dev_preds_ridgeCV, best)

print(best)
print("(Best) RMSLE for ENSEMBLE on dev set:", rmsle2(Y_dev, Y_dev_preds))

# best predicted submission
preds = aggregate_predicts2(rnn_preds, ridgeCV_preds, best)
submission = pd.DataFrame({
        "test_id": test_df.test_id,
        "price": preds.reshape(-1),
})
#submission.to_csv("./rnn_ridge_submission_best.csv", index=False)
submission.to_csv(path+"RNN_ridgeCV_submission_best.csv", index=False)

# mean submission
preds = aggregate_predicts2(rnn_preds, ridgeCV_preds, 0.5)
submission = pd.DataFrame({
        "test_id": test_df.test_id,
        "price": preds.reshape(-1),
})
#submission.to_csv("./rnn_ridge_submission_mean.csv", index=False)
submission.to_csv(path+"RNN_ridgeCV_submission_mean.csv", index=False)

# variation 1 submission
preds = aggregate_predicts2(rnn_preds, ridgeCV_preds, best+0.1)
submission = pd.DataFrame({
        "test_id": test_df.test_id,
        "price": preds.reshape(-1),
})
#submission.to_csv("./rnn_ridge_submission_var_1.csv", index=False)
submission.to_csv(path+"RNN_ridgeCV_submission_var1.csv", index=False)



