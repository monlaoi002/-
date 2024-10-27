#일단 이게 가장 좋은거같은
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

namesss = 'AMZN'

# 데이터 다운로드 및 준비
df = yf.download(namesss, start='2010-01-01', end='2024-10-23')

# 기존 및 새로운 기술적 지표 계산 함수들
def compute_rsi(data, window=14):
    diff = data.diff()
    up = diff.clip(lower=0)
    down = -1 * diff.clip(upper=0)
    ema_up = up.ewm(com=window-1, adjust=False).mean()
    ema_down = down.ewm(com=window-1, adjust=False).mean()
    rs = ema_up / ema_down
    return 100 - (100 / (1 + rs))

def compute_macd(data, fast=12, slow=26, signal=9):
    ema_fast = data.ewm(span=fast, adjust=False).mean()
    ema_slow = data.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def compute_bollinger_bands(data, window=20, num_std=2):
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

def compute_stochastic_oscillator(data, k_window=14, d_window=3):
    low_min = data['Low'].rolling(window=k_window).min()
    high_max = data['High'].rolling(window=k_window).max()
    k = 100 * (data['Close'] - low_min) / (high_max - low_min)
    d = k.rolling(window=d_window).mean()
    return k, d

def compute_atr(data, window=14):
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(window=window).mean()

def compute_obv(data):
    obv = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()
    return obv

def compute_cci(data, window=20):
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    sma = typical_price.rolling(window=window).mean()
    mad = typical_price.rolling(window=window).apply(lambda x: np.abs(x - x.mean()).mean())
    cci = (typical_price - sma) / (0.015 * mad)
    return cci

# 모든 지표 계산 및 데이터프레임에 추가
df['MA5'] = df['Close'].rolling(window=5).mean()
df['MA20'] = df['Close'].rolling(window=20).mean()
df['RSI'] = compute_rsi(df['Close'])
df['MACD'], df['Signal_Line'] = compute_macd(df['Close'])
df['Upper_BB'], df['Lower_BB'] = compute_bollinger_bands(df['Close'])
df['Stoch_K'], df['Stoch_D'] = compute_stochastic_oscillator(df)
df['ATR'] = compute_atr(df)
df['OBV'] = compute_obv(df)
df['CCI'] = compute_cci(df)

# 결측치 제거
df.dropna(inplace=True)

# 타겟 변수 생성 (다음 날 종가)
df['Target'] = df['Close'].shift(-1)
df = df[:-1]  # 마지막 행 제거 (Target이 NaN)

# 특성과 타겟 분리
feature_columns = ['Close', 'MA5', 'MA20', 'RSI', 'MACD', 'Signal_Line', 'Upper_BB', 'Lower_BB', 
                   'Stoch_K', 'Stoch_D', 'ATR', 'OBV', 'CCI']
X = df[feature_columns]
y = df['Target']

# 데이터 전처리
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 모델 생성 및 훈련
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 예측 및 평가
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")
'''
# 결과 시각화
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(f"Actual vs Predicted {namesss} Prices")
plt.show()

# 특징 중요도 시각화
feature_importance = pd.DataFrame({'feature': feature_columns, 'importance': model.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False)

plt.figure(figsize=(12, 6))
plt.bar(feature_importance['feature'], feature_importance['importance'])
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=90, ha='right')
plt.tight_layout()
plt.show()
'''
# 미래 가격 예측
last_data = X_scaled[-1].reshape(1, -1)
next_day_prediction = model.predict(last_data)

print(f"Predicted price for next day: {next_day_prediction[0]}")