import numpy as np, joblib
from tensorflow.keras import layers, models
from utils.data_preprocess import load_and_prepare_data

(X_train, y_train, X_val, y_val, X_test, y_test, features, scaler_y) = load_and_prepare_data()

mean_importance = joblib.load("month_weights.pkl")
weights = mean_importance / mean_importance.sum()

review_idx = [features.index(f) for f in [
    'review_title_tfidf','review_text_tfidf',
    'review_title_sentiment','review_text_sentiment'
]]

seq_len = X_train.shape[1]
def apply_shap_weights(X):
    X_w = X.copy()
    for t in range(seq_len):
        X_w[:, t, review_idx] *= weights[t]
    return X_w

X_train_w, X_val_w, X_test_w = map(apply_shap_weights, [X_train, X_val, X_test])

model_w = models.Sequential([
    layers.Input(shape=(seq_len, X_train.shape[2])),
    layers.LSTM(64),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])
model_w.compile(optimizer='adam', loss='mse')
model_w.fit(X_train_w, y_train, validation_data=(X_val_w, y_val),
            epochs=100, batch_size=8, verbose=1)

y_pred_scaled = model_w.predict(X_test_w)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_test)
print("Predicted vs True sample:\n", np.hstack([y_pred[:5], y_true[:5]]))
