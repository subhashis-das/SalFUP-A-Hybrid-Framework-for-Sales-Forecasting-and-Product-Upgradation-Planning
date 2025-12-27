# """
# Train a SentiMixer model directly from a preprocessed CSV file.

# Usage:
#     python scripts/train_sentitsmixer.py <path_to_csv> [output_folder]

# Example:
#     python scripts/train_sentitsmixer.py data/monthly_grouped_shap.csv results/run_sarima_1
# """
# import os
# import sys
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error
# from tensorflow.keras import layers

# # --- Arguments ---
# if len(sys.argv) < 2:
#     print("Usage: python train_sarima.py <csv_path> [output_folder]")
#     sys.exit(1)

# csv_path = sys.argv[1]
# output_folder = sys.argv[2] if len(sys.argv) > 2 else "results/sarima"
# os.makedirs(output_folder, exist_ok=True)

# # --- Load data ---
# df = pd.read_csv(csv_path)
# df['month'] = pd.to_datetime(df['month'])
# df.set_index('month', inplace=True)

# df_grouped = df
# df_grouped['average_sentiment'] = (df['review_text_sentiment']+df['review_title_sentiment']) /2

# min_sentiment=df_grouped['average_sentiment'].min()
# max_sentiment=df_grouped['average_sentiment'].max()


# df_grouped['average_sentiment']=1+(df_grouped['average_sentiment']-min_sentiment)*(5-1)/(max_sentiment-min_sentiment)

# max_value = df_grouped['helpful_vote'].max()
# min_value = df_grouped['helpful_vote'].min()

# df_grouped['helpful_vote']=1+ (df_grouped['helpful_vote']-min_value)*(4)/(max_value-min_value)
# df_grouped['mixture']=df_grouped['helpful_vote']*df_grouped['average_sentiment']*df_grouped['rating']
# df_grouped['mixture']=1+ (df_grouped['mixture']-min_value)*(4)/(max_value-min_value)


# max_value = df_grouped['sales'].max()
# min_value = df_grouped['sales'].min()

# print(min_value,max_value)
# df_grouped['sales']=1+ (df_grouped['sales']-min_value)*(4)/(max_value-min_value)

# df_grouped=df_grouped.drop(['review_title_sentiment','review_text_sentiment'],axis=1)

# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# import matplotlib.pyplot as plt

# # Assuming df_grouped is your DataFrame and 'review_count' is your target
# X = df_grouped[['rating', 'helpful_vote', 'verified_purchase', 'review_title_tfidf','review_text_tfidf', 'average_sentiment', 'mixture']]
# y = df_grouped['sales']

# # Create a RandomForestRegressor model
# model = RandomForestRegressor(n_estimators=100, random_state=42)

# # Fit the model
# model.fit(X, y)

# # Get feature importances
# importances = model.feature_importances_

# # Create a DataFrame for better visualization
# importance_df = pd.DataFrame({
#     'Feature': X.columns,
#     'Importance': importances
# })

# # Sort the features by importance
# importance_df = importance_df.sort_values(by='Importance', ascending=False)

# # Display the feature importances
# print(importance_df)

# #Plot the feature importances
# importance_df.plot(kind='bar', x='Feature', y='Importance', legend=False, title="Feature Importances")
# # plt.show()


# feature_importance = dict(zip(importance_df['Feature'], importance_df['Importance']))

# # Normalize the feature importance to sum to 1
# total_importance = sum(feature_importance.values())

# normalized_importance = {key: value / total_importance for key, value in feature_importance.items()}

# # Scaling the relevant columns based on the normalized feature importance
# df_scaled = df_grouped.copy()
# df_scaled['helpful_vote'] *= normalized_importance['helpful_vote']
# df_scaled['rating'] *= normalized_importance['rating']
# df_scaled['average_sentiment'] *= normalized_importance['average_sentiment']


# #  split train/valid/test
# n = len(df_grouped)
# train_end = int(n * 0.7)
# val_end = n - int(n * 0.1)
# test_end = n

# seq_len=16
# pred_len=16
# batch_size=60

# df_grouped_bak=df_grouped

# df_grouped_bak['sales'] = np.log(df_grouped_bak['sales']+1)

# log_min = df_grouped_bak['sales'].min()
# log_max = df_grouped_bak['sales'].max()


# df_grouped_bak=df_grouped_bak/10


# train_df = df_grouped_bak[:train_end]
# val_df = df_grouped_bak[train_end - seq_len: val_end]
# test_df = df_grouped_bak[val_end - seq_len: test_end]


# def _split_window(data,target_slice=slice(0,None)):
#     inputs = data[:, : seq_len, :]
#     labels = data[:, seq_len :, target_slice]
#     inputs.set_shape([None, seq_len, None])
#     labels.set_shape([None, pred_len, None])
#     return inputs, labels

# def _make_dataset(data, shuffle=True, targets=None):
#     data = np.array(data, dtype=np.float32)
#     ds = tf.keras.utils.timeseries_dataset_from_array(
#         data=data,
#         targets=targets,
#         sequence_length=(seq_len + pred_len),
#         sequence_stride=1,
#         shuffle=shuffle,
#         batch_size=batch_size,
#     )
#     ds = ds.map(_split_window)
#     return ds

# def get_train(self, shuffle=True):
#     return self._make_dataset(self.train_df, shuffle=shuffle)

# import tensorflow as tf
# train_data  =   _make_dataset(train_df, shuffle=False)
# val_data    =   _make_dataset(val_df, shuffle=False)
# test_data   =   _make_dataset(test_df, shuffle=False)


# print("len(test_df):", len(test_df))
# print("seq_len:", seq_len)
# print("pred_len:", pred_len)
# print("required minimum:", seq_len + pred_len)




# def res_block(inputs, ff_dim):
#     norm = layers.LayerNormalization

#     # Time mixing
#     x = norm(axis=[-2, -1])(inputs)
#     x = layers.Permute((2, 1))(x)            # [B, C, T]
#     x = layers.Dense(x.shape[-1], activation="relu")(x)
#     x = layers.Permute((2, 1))(x)            # [B, T, C]
#     x = layers.Dropout(0.6)(x)
#     res = layers.Add()([x, inputs])

#     # Feature mixing
#     x = norm(axis=[-2, -1])(res)
#     x = layers.Dense(ff_dim, activation="relu")(x)
#     x = layers.Dropout(0.6)(x)
#     x = layers.Dense(inputs.shape[-1])(x)
#     x = layers.Dropout(0.6)(x)

#     return layers.Add()([x, res])

# def build_model(
#     input_shape,
#     pred_len,
#     n_block,
#     ff_dim,
#     target_slice,
# ):
#     inputs = layers.Input(shape=input_shape)
#     x = inputs

#     for _ in range(n_block):
#         x = res_block(x, ff_dim)

#     if target_slice:
#         x = x[:, :, target_slice]

#     # Temporal projection
#     x = layers.Permute((2, 1))(x)     # [B, C, T]
#     x = layers.Dense(pred_len)(x)     # [B, C, O]
#     outputs = layers.Permute((2, 1))(x)  # [B, O, C]

#     return tf.keras.Model(inputs, outputs)


# n_feature = train_df.shape[-1]
# model = build_model(
#     input_shape=(seq_len, n_feature),
#     pred_len=pred_len,
#     n_block=16,
#     ff_dim=64,
#     target_slice=slice(0,None)
# )


# from tensorflow.keras.callbacks import ModelCheckpoint
# tf.keras.utils.set_random_seed(7)

# initial_learning_rate = 0.1
# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate,
#     decay_steps=24,
#     decay_rate=0.97,
#     staircase=True
# )

# optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# model.compile(optimizer, loss='mse', metrics=['mse'])
# checkpoint_callback = ModelCheckpoint(
#     filepath="best_model.keras",
#     monitor="val_mse",
#     save_best_only=True,
#     mode="min",
#     verbose=1,
# )

# history = model.fit(
#     train_data,
#     epochs= 1000,
#     validation_data=val_data,
#     callbacks=[checkpoint_callback]
# )

# model = tf.keras.models.load_model("best_model.keras")
# predictions = model.predict(test_data)


# cols = df_grouped.columns
# cols

# scaled_preds = predictions[-1,:,:]

# scaled_preds.shape

# scaled_preds_df = pd.DataFrame(scaled_preds)
# scaled_preds_df.columns = cols

# cols =['rating','mixture','sales','helpful_vote','average_sentiment']


# fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12,8))

# for i, ax in enumerate(axes.flatten()):
#     col = cols[i+1]
#     print(col)
#     ax.plot(df_grouped.index[-pred_len:], df_grouped_bak[col][-pred_len:], label = 'Actual Value', color='blue')
#     ax.plot(df_grouped.index[-pred_len:], scaled_preds_df[col], label='TSMixer', ls='--', color='red')
    
#     ax.legend(loc='best')
#     ax.set_xlabel('Time steps')
#     ax.set_ylabel('Value')
#     ax.set_ylim(-2, 2) 
#     ax.set_title(col)
#     for column in cols[1:5]:
#         plt.tight_layout()
#         fig.autofmt_xdate()
# plt.savefig(os.path.join(output_folder, "SentiTSMixer.png"), dpi=200)

# metrics_path = os.path.join(output_folder, "metrics.txt")
# with open(metrics_path, "a") as f:
#     f.write(f"SentiTSMixer \n")
#     for column in df_grouped.columns:
#         mse = ((df_grouped_bak[column][-pred_len:].values - scaled_preds_df[column].values) ** 2).mean()
#         msle = ((np.log1p(df_grouped_bak[column][-pred_len:].values) - np.log1p(scaled_preds_df[column].values)) ** 2).mean()
#         mape= (abs((df_grouped_bak[column][-pred_len:].values - scaled_preds_df[column].values) / df_grouped_bak[column][-pred_len:].values)).mean()
#         f.write(f"MSE({column})={str(round(mse,5))}\n")
#         f.write(f"MSLE({column})={str(round(msle,5))}\n")
#         f.write(f"MAPE({column})={str(round(mape,5))}\n")

# print("Saved forecast plot and results of SENTITSMIXER")


"""
Train a SentiMixer model directly from a preprocessed CSV file
and save results in the same folder structure as ARIMA outputs.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras import layers
import tensorflow as tf

# --- Arguments ---
if len(sys.argv) < 2:
    print("Usage: python train_sentitsmixer.py <csv_path> [output_folder]")
    sys.exit(1)

csv_path = sys.argv[1]
output_folder = sys.argv[2] if len(sys.argv) > 2 else "results/sarima"
os.makedirs(output_folder, exist_ok=True)

# --- Load and preprocess data ---
df = pd.read_csv(csv_path)
df['month'] = pd.to_datetime(df['month'])
df.set_index('month', inplace=True)

df_grouped = df.copy()
df_grouped['average_sentiment'] = (df['review_text_sentiment'] + df['review_title_sentiment']) / 2

# Scale average_sentiment to ARIMA range (1-5)
min_sentiment = df_grouped['average_sentiment'].min()
max_sentiment = df_grouped['average_sentiment'].max()
df_grouped['average_sentiment'] = 1 + (df_grouped['average_sentiment'] - min_sentiment) * (5 - 1) / (max_sentiment - min_sentiment)

# Scale helpful_vote to ARIMA range (1-5)
min_helpful = df_grouped['helpful_vote'].min()
max_helpful = df_grouped['helpful_vote'].max()
df_grouped['helpful_vote'] = 1 + (df_grouped['helpful_vote'] - min_helpful) * 4 / (max_helpful - min_helpful)

# Mixture feature
df_grouped['mixture'] = df_grouped['helpful_vote'] * df_grouped['average_sentiment'] * df_grouped['rating']
df_grouped['mixture'] = 1 + (df_grouped['mixture'] - min_helpful) * 4 / (max_helpful - min_helpful)

# Scale sales to ARIMA range (1-5)
min_sales = df_grouped['sales'].min()
max_sales = df_grouped['sales'].max()
df_grouped['sales'] = 1 + (df_grouped['sales'] - min_sales) * 4 / (max_sales - min_sales)

# Drop unnecessary columns
df_grouped = df_grouped.drop(['review_title_sentiment', 'review_text_sentiment'], axis=1)

# --- Feature importance based scaling ---
X_rf = df_grouped[['rating', 'helpful_vote', 'verified_purchase', 'review_title_tfidf', 'review_text_tfidf', 'average_sentiment', 'mixture']]
y_rf = df_grouped['sales']

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_rf, y_rf)

importances = rf.feature_importances_
importance_df = pd.DataFrame({'Feature': X_rf.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)
feature_importance = dict(zip(importance_df['Feature'], importance_df['Importance']))
total_importance = sum(feature_importance.values())
normalized_importance = {k: v / total_importance for k, v in feature_importance.items()}

# Scale selected features by importance
df_scaled = df_grouped.copy()
df_scaled['helpful_vote'] *= normalized_importance['helpful_vote']
df_scaled['rating'] *= normalized_importance['rating']
df_scaled['average_sentiment'] *= normalized_importance['average_sentiment']

# --- Train/val/test split ---
n = len(df_grouped)
train_end = int(n * 0.7)
val_end = n - int(n * 0.1)
test_end = n

seq_len = 16
pred_len = 16
batch_size = 60

df_grouped_bak = df_grouped.copy()
df_grouped_bak['sales'] = np.log(df_grouped_bak['sales'] + 1)
df_grouped_bak /= 10  # consistent scaling

train_df = df_grouped_bak[:train_end]
val_df = df_grouped_bak[train_end - seq_len: val_end]
test_df = df_grouped_bak[val_end - seq_len: test_end]

# --- Dataset utils ---
def _split_window(data, target_slice=slice(0, None)):
    inputs = data[:, :seq_len, :]
    labels = data[:, seq_len:, target_slice]
    inputs.set_shape([None, seq_len, None])
    labels.set_shape([None, pred_len, None])
    return inputs, labels

def _make_dataset(data, shuffle=True):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.utils.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=seq_len + pred_len,
        sequence_stride=1,
        shuffle=shuffle,
        batch_size=batch_size,
    )
    ds = ds.map(_split_window)
    return ds

train_data = _make_dataset(train_df, shuffle=False)
val_data = _make_dataset(val_df, shuffle=False)
test_data = _make_dataset(test_df, shuffle=False)

# --- TSMixer model ---
def res_block(inputs, ff_dim):
    x = layers.LayerNormalization(axis=[-2, -1])(inputs)
    x = layers.Permute((2, 1))(x)
    x = layers.Dense(x.shape[-1], activation="relu")(x)
    x = layers.Permute((2, 1))(x)
    x = layers.Dropout(0.6)(x)
    res = layers.Add()([x, inputs])

    x = layers.LayerNormalization(axis=[-2, -1])(res)
    x = layers.Dense(ff_dim, activation="relu")(x)
    x = layers.Dropout(0.6)(x)
    x = layers.Dense(inputs.shape[-1])(x)
    x = layers.Dropout(0.6)(x)
    return layers.Add()([x, res])

def build_model(input_shape, pred_len, n_block, ff_dim, target_slice):
    inputs = layers.Input(shape=input_shape)
    x = inputs
    for _ in range(n_block):
        x = res_block(x, ff_dim)
    if target_slice:
        x = x[:, :, target_slice]
    x = layers.Permute((2, 1))(x)
    x = layers.Dense(pred_len)(x)
    outputs = layers.Permute((2, 1))(x)
    return tf.keras.Model(inputs, outputs)

n_feature = train_df.shape[-1]
model = build_model(
    input_shape=(seq_len, n_feature),
    pred_len=pred_len,
    n_block=16,
    ff_dim=64,
    target_slice=slice(0, None)
)

# --- Compile & train ---
tf.keras.utils.set_random_seed(7)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.1,
    decay_steps=24,
    decay_rate=0.97,
    staircase=True
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(optimizer, loss='mse', metrics=['mse'])

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(output_folder, "best_model.keras"),
    monitor="val_mse",
    save_best_only=True,
    mode="min",
    verbose=1,
)

history = model.fit(
    train_data,
    epochs=100,
    validation_data=val_data,
    callbacks=[checkpoint_callback]
)

model = tf.keras.models.load_model(os.path.join(output_folder, "best_model.keras"))
predictions = model.predict(test_data)

# --- Save plots & metrics ---
cols = ['rating', 'mixture', 'sales', 'helpful_vote', 'average_sentiment']
scaled_preds = predictions[-1, :, :]
scaled_preds_df = pd.DataFrame(scaled_preds, columns=df_grouped.columns)

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12, 8))
for i, ax in enumerate(axes.flatten()):
    col = cols[i+1]
    ax.plot(df_grouped.index[-pred_len:], df_grouped_bak[col][-pred_len:], label='Actual', color='blue')
    ax.plot(df_grouped.index[-pred_len:], scaled_preds_df[col], label='TSMixer', ls='--', color='red')
    ax.legend(loc='best')
    ax.set_xlabel('Time steps')
    ax.set_ylabel('Value')
    ax.set_ylim(-2, 2)
    ax.set_title(col)
plt.tight_layout()
fig.autofmt_xdate()
plt.savefig(os.path.join(output_folder, "SentiTSMixer.png"), dpi=200)

metrics_path = os.path.join(output_folder, "metrics.txt")
with open(metrics_path, "a") as f:
    f.write(f"\n______________________________________\nSentiTSMixer: \n")
    for column in df_grouped.columns:
        mse = ((df_grouped_bak[column][-pred_len:].values - scaled_preds_df[column].values) ** 2).mean()
        msle = ((np.log1p(df_grouped_bak[column][-pred_len:].values) - np.log1p(scaled_preds_df[column].values)) ** 2).mean()
        mape = (abs((df_grouped_bak[column][-pred_len:].values - scaled_preds_df[column].values) / df_grouped_bak[column][-pred_len:].values)).mean()
        f.write(f"MSE({column})={round(mse,5)}\n")
        f.write(f"MSLE({column})={round(msle,5)}\n")
        f.write(f"MAPE({column})={round(mape,5)}\n")

print("Saved forecast plot and results of SentiTSMixer in", output_folder)
