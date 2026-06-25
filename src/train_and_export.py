"""
Train and Export Cascade B & RAG Artifacts for Production
=========================================================
"""

import sys
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier

TRAIN_PATH = "data/split/train.csv"

def make_preprocessor(num_cols, cat_cols):
    nt = Pipeline([('imp', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    ct = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                   ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    return ColumnTransformer([('n', nt, num_cols), ('c', ct, cat_cols)])

def main():
    print("Loading datasets...")
    df = pd.read_csv(TRAIN_PATH, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print("Building historical feature base...")
    df['log_amount'] = np.log1p(df['amount'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
    df['is_night'] = ((df['hour']>=0)&(df['hour']<=5)).astype(int)
    df['tslt_abs'] = df['time_since_last_transaction'].abs()
    df['tslt_is_missing'] = df['time_since_last_transaction'].isnull().astype(int)
    df['tslt_is_negative'] = (df['time_since_last_transaction']<0).astype(int)
    
    pairs = [
        ('sender_account', 'location', 'sender_location_pair_count_past', 'is_new_location_for_sender'),
        ('sender_account', 'payment_channel', 'sender_channel_pair_count_past', 'is_new_payment_channel_for_sender'),
        ('sender_account', 'device_used', 'sender_device_type_pair_count_past', 'is_new_device_used_for_sender'),
        ('sender_account', 'transaction_type', 'sender_txn_type_pair_count_past', 'is_new_transaction_type_for_sender'),
    ]
    for s, v, cnt_f, nov_f in pairs:
        df2 = df[[s, v, 'timestamp']].copy()
        df2['oi'] = df2.index
        df2 = df2.sort_values([s, v, 'timestamp'])
        df2['pc'] = df2.groupby([s, v]).cumcount()
        df2 = df2.sort_values('oi')
        df[nov_f] = (df2['pc'].values == 0).astype(int)

    NUM_FS1 = ['amount','log_amount','spending_deviation_score','velocity_score',
               'geo_anomaly_score','hour','day_of_week','month','is_weekend','is_night',
               'tslt_abs','tslt_is_missing','tslt_is_negative',
               'is_new_location_for_sender','is_new_payment_channel_for_sender',
               'is_new_transaction_type_for_sender','is_new_device_used_for_sender']
    NUM_HARD = [c for c in NUM_FS1 if c != 'tslt_is_missing']
    CAT_COLS = ['transaction_type','merchant_category','payment_channel','device_used','location']
    
    # Use the same split logic
    train_inner = df.iloc[:3200000]
    
    print("Training M1 Calibrated (Cascade B)...")
    # For speed of export in prototype, we'll use a sample if needed, but 3.2M is fast enough for LightGBM
    # Let's train on last 1M rows of inner to speed up the calibration CV without losing too much performance
    train_sample = train_inner.iloc[-1000000:]
    y_tr = train_sample['is_fraud'].astype(int)
    h_tr = train_sample['tslt_is_missing'] == 0
    X_tr_h = train_sample[h_tr]
    y_tr_h = y_tr[h_tr]
    
    pre = make_preprocessor(NUM_HARD, CAT_COLS)
    X_tr_h_trans = pre.fit_transform(X_tr_h[NUM_HARD+CAT_COLS])
    
    lgbm_config = dict(n_estimators=100, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1)
    cal_mdl = CalibratedClassifierCV(estimator=LGBMClassifier(**lgbm_config), method='sigmoid', cv=3)
    cal_mdl.fit(X_tr_h_trans, y_tr_h)
    
    with open("models/preprocessor.pkl", "wb") as f:
        pickle.dump(pre, f)
    with open("models/calibrated_lgbm.pkl", "wb") as f:
        pickle.dump(cal_mdl, f)
        
    print("Building RAG Retriever Corpus...")
    corpus_idx = train_sample.groupby('is_fraud').apply(lambda x: x.sample(3000, random_state=42)).index.get_level_values(1)
    corpus_df = train_sample.loc[corpus_idx]
    corpus_trans = pre.transform(corpus_df[NUM_HARD+CAT_COLS])
    
    retriever = NearestNeighbors(n_neighbors=3, algorithm='auto')
    retriever.fit(corpus_trans)
    
    with open("models/retriever.pkl", "wb") as f:
        pickle.dump(retriever, f)
    # Save the corpus df to lookup labels and text
    corpus_df.to_pickle("models/rag_corpus.pkl")
    
    print("Exporting Pseudo-Feature Store (In-Memory History)...")
    # Save a lightweight history of pairs to compute novelty for new incoming transactions
    # To keep RAM small, we only save the last 200K unique pairs
    history_df = df.iloc[-500000:][['sender_account', 'location', 'payment_channel', 'device_used', 'transaction_type']]
    history_df.to_pickle("models/history_store.pkl")

    print("Export Complete! All artifacts saved to models/")

if __name__ == "__main__":
    main()
