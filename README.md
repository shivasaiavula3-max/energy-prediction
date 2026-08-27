
# ⚡ Energy Consumption Prediction  
A complete end‑to‑end workflow for predicting household appliance energy usage using **Machine Learning**, **Deep Learning**, **Transformers**, and **Explainability (SHAP)**.

---

## 📁 Project Structure

Create a folder: Dissertation,

Place your dataset inside the folder:energydata_complete.csv \url{https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction}

Open **Jupyter Notebook**, navigate into the folder, and create:energy_prediction.ipynb

---

## 1. Import Core Libraries

Your notebook begins by importing all required libraries:

- Python  
- Pandas  
- NumPy  
- scikit‑learn  
- Matplotlib  
- PyTorch  
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import torch
```

These handle data loading, preprocessing, ML models, and deep learning.


## 2. Load Dataset

Load the dataset and print:

- shape  
- columns  
- first rows  
- datatypes  
- missing values  
- date range  
- target statistics
 ``` 
df = pd.read_csv('energydata_complete.csv')
print(df.head())
print(df.info())
```


This confirms the dataset is correct.


## 3. Convert Date Column

Convert the `date` column to datetime and set it as the index for time‑series modeling.
```
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

```
---

## 4. Plot Target Variable

Plot:

- Full appliance energy usage  
- A sample week
```
df['Appliances'].plot(figsize=(14,4))
plt.title("Appliance Energy Usage")
plt.show()

```

This helps visualize consumption patterns.

---

## 5. Feature Engineering

Add:

- hour, day_of_week, weekend, month
```
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek  # 0=Monday
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['month'] = df.index.month
``` 
- cyclical hour encoding
```
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
``` 
- lag features
```
for lag in [1, 2, 3, 6, 12, 24, 48]:  # 10min, 20min, 30min, 1h, 2h, 4h, 8h, 24h
  df[f'lag_{lag}'] = df['Appliances'].shift(lag)

```  
- rolling mean & std
```
df['rolling_mean_6'] = df['Appliances'].shift(1).rolling(6).mean()   # 1h
df['rolling_mean_24'] = df['Appliances'].shift(1).rolling(24).mean() # 4h
df['rolling_std_6'] = df['Appliances'].shift(1).rolling(6).std()
```  
- temperature–humidity interaction
```
df['T_RH_interaction'] = df['T_out'] * df['RH_out']
```

Drop NaN rows created by lag/rolling.
```
df_features = df_features.dropna()
```
---

## 6. Train / Validation / Test Split

Split the dataset chronologically:

- 70% training  
- 15% validation  
- 15% testing
```
n = len(df_features)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train = df_features.iloc[:train_end]
val = df_features.iloc[train_end:val_end]
test = df_features.iloc[val_end:]
```

Print and Plot the split to verify.

---

## 7. Scaling

Scale features using `StandardScaler`:

- Fit on **train only**  
- Transform validation and test
```
feature_cols = [c for c in df_features.columns if c != 'Appliances']
target_col = 'Appliances'
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train = scaler_X.fit_transform(train[feature_cols])
y_train = scaler_y.fit_transform(train[[target_col]]).ravel()
X_val = scaler_X.transform(val[feature_cols])
y_val = scaler_y.transform(val[[target_col]]).ravel()
X_test = scaler_X.transform(test[feature_cols])
y_test = scaler_y.transform(test[[target_col]]).ravel()

``` 

Save the dataframes:
data/train_scaled.csv

data/val_scaled.csv

data/test_scaled.csv

models/scaler_X.pkl

models/scaler_y.pkl


---

## 8. Machine Learning Models

Train:

- Linear Regression
``` 
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
``` 
- Random Forest
```
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=800, max_depth=None,min_samples_split=4,min_samples_leaf=2,max_features='sqrt', n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
```  
- XGBoost  
```
xgb = XGBRegressor(
    n_estimators=1500,
    max_depth=6,
    learning_rate=0.02,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=2,
    reg_alpha=0.5,
    n_jobs=-1,
    random_state=42,
    early_stopping_rounds=20   
)
xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
```  
Evaluate using:

- MAE 
- RMSE 
- R²
- MAPE
```
mae  = mean_absolute_error(y_true,  y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2   = r2_score(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100
```  

Save outputs:
outputs/baseline_results.csv

outputs/xgb_feature_importance.png


---

## 9. Deep Learning Models (PyTorch)

Models:

- **LSTM**
```
class LSTMModel(nn.Module):
    def __init__(self, n_features, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.drop(out[:, -1, :])
        return self.head(out).squeeze(-1)
```  
- **TCN**
```
class TCNModel(nn.Module):
    def __init__(self, n_features, channels=16, kernel=3, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, channels, kernel, padding=kernel-1)
        self.conv2 = nn.Conv1d(channels, channels, kernel, padding=kernel-1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(channels, 8), nn.ReLU(), nn.Linear(8, 1))
    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x)[:, :, :x.size(2)])
        x = self.drop(x)
        x = torch.relu(self.conv2(x)[:, :, :x.size(2)])
        return self.head(x[:, :, -1]).squeeze(-1)
```  

Includes:

- SequenceDataset  
- Early stopping  
- Gradient clipping  
- Training curves  

Save:
models/lstm.pt

models/tcn.pt

outputs/deep_learning_results.csv


---

## 10. Transformer Model + Attention

Includes:

- Positional encoding
```
class TransformerModel(nn.Module):
    def __init__(self, n_features, d_model=32, nhead=4, layers=2, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, LOOKBACK, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1)
        )
        self.last_attn = None

    def forward(self, x, capture_attn=False):
        x = self.input_proj(x) + self.pos_encoder
        for layer in self.transformer.layers:
            if capture_attn:
                attn_out, attn_weights = layer.self_attn(
                    x, x, x, need_weights=True, average_attn_weights=True
                )
                self.last_attn = attn_weights
                x = x + layer.dropout1(attn_out)
                x = layer.norm1(x)
                ff_out = layer.linear2(layer.dropout(torch.relu(layer.linear1(x))))
                x = x + layer.dropout2(ff_out)
                x = layer.norm2(x)
            else:
                x = layer(x)
        return self.head(x[:, -1, :]).squeeze(-1)
```  
- Multi‑head self‑attention 
- Attention extraction
```
transformer.eval()
sample_x, sample_y = next(iter(test_loader))
sample_x = sample_x.to(device)

with torch.no_grad():
    _ = transformer(sample_x, capture_attn=True)
    attn = transformer.last_attn.cpu().numpy()

# Average attention across batch
avg_attn = attn.mean(axis=0)
```  



Save:
models/transformer.pt

outputs/transformer_attention_heatmap.png

outputs/transformer_attention_bars.png


---

## 11. SHAP Explainability

Generate:

- SHAP summary plot
```
explainer = shap.LinearExplainer(lr, X_train)
shap_values = explainer(X_test)

# SHAP Summary Plot (beeswarm — shows direction of impact)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values.values, X_test, feature_names=feature_cols, show=False)
```  
- SHAP bar plot
```
shap.summary_plot(shap_values.values, X_test, feature_names=feature_cols, plot_type='bar', show=False)
```  
- SHAP force plot for peak usage
```
peak_idx = y_test.idxmax()
peak_row = X_test.loc[[peak_idx]]
peak_shap = explainer(peak_row)

# Force plot for this single instance
plt.figure(figsize=(12, 3))
shap.force_plot(explainer.expected_value, peak_shap.values[0], peak_row,
                feature_names=feature_cols, matplotlib=True, show=False)
```   

Save:
outputs/shap_summary.png

outputs/shap_bar.png

outputs/shap_force_peak.png

outputs/shap_values.npy


---

## Final Output Summary

Your workflow produces:

- Preprocessed datasets  
- ML baseline models  
- LSTM, TCN, Transformer models  
- Attention heatmaps  
- SHAP explainability  
- All results saved in `outputs/`  

---






