#!/usr/bin/env python3
"""
AI Stock Trading Analysis Dashboard
Ensemble ML (RandomForest + GradientBoosting) + Deep Learning (LSTM)
Built for Streamlit Cloud - Including Full ML & DL Multi-Ticker Scan
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
import warnings

warnings.filterwarnings("ignore")

# Streamlit Pagina Configuratie
st.set_page_config(page_title="AI Stock Trading Dashboard", layout="wide")

# -----------------------
# Technical Indicators
# -----------------------
def add_technical_indicators(df):
    df = df.copy()
    # Trend
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
    df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)
    df['MACD'] = ta.trend.macd(df['Close'])
    df['MACD_signal'] = ta.trend.macd_signal(df['Close'])
    df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14)
    
    # Momentum
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['Stoch'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
    df['CCI'] = ta.trend.cci(df['High'], df['Low'], df['Close'], window=20)
    
    # Volatility
    df['BB_high'] = ta.volatility.bollinger_hband(df['Close'])
    df['BB_low'] = ta.volatility.bollinger_lband(df['Close'])
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
    
    # Volume
    df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    df['MFI'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'])
    
    # Returns & targets
    df['Return'] = df['Close'].pct_change()
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)  # Next day up/down
    
    df = df.dropna()
    return df

# -----------------------
# Core Ensemble ML Engine
# -----------------------
def calc_ensemble(df):
    feature_cols = [
        'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'MACD', 'MACD_signal', 'ADX',
        'RSI', 'Stoch', 'CCI', 'BB_high', 'BB_low', 'ATR', 'OBV', 'MFI', 'Return'
    ]
    available = [c for c in feature_cols if c in df.columns]
    X = df[available]
    y = df['Target']
    
    if len(X) < 80:
        return None
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    gb.fit(X_train, y_train)
    gb_pred = gb.predict(X_test)
    gb_acc = accuracy_score(y_test, gb_pred)
    
    rf_proba = rf.predict_proba(X_test)[:, 1]
    gb_proba = gb.predict_proba(X_test)[:, 1]
    ensemble_proba = (rf_proba + gb_proba) / 2
    ensemble_pred = (ensemble_proba > 0.5).astype(int)
    ens_acc = accuracy_score(y_test, ensemble_pred)
    
    latest_features = X.iloc[[-1]]
    rf_latest = rf.predict_proba(latest_features)[0, 1]
    gb_latest = gb.predict_proba(latest_features)[0, 1]
    ens_latest = (rf_latest + gb_latest) / 2
    
    rf_imp = pd.Series(rf.feature_importances_, index=available)
    gb_imp = pd.Series(gb.feature_importances_, index=available)
    avg_imp = ((rf_imp + gb_imp) / 2).sort_values(ascending=False).head(10)
    
    return {
        'rf_acc': rf_acc,
        'gb_acc': gb_acc,
        'ens_acc': ens_acc,
        'prob_up': ens_latest,
        'ensemble_proba': ensemble_proba,
        'X_test': X_test,
        'feature_imp': avg_imp
    }

# -----------------------
# Core LSTM Deep Learning Engine
# -----------------------
class StockDataset(Dataset):
    def __init__(self, data, seq_len=30):
        self.data = data
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y = self.data[idx+self.seq_len]
        return torch.FloatTensor(x), torch.FloatTensor([y])

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def calc_lstm(df, epochs=10, seq_len=30):
    if len(df) < 100:
        return None
        
    prices = df['Close'].values.astype(np.float32)
    train_size = int(len(prices) * 0.8)
    
    scaler = StandardScaler()
    train_prices = prices[:train_size].reshape(-1, 1)
    scaler.fit(train_prices)
    
    prices_scaled = scaler.transform(prices.reshape(-1, 1)).flatten()
    
    train_data = prices_scaled[:train_size]
    test_data = prices_scaled[train_size - seq_len:]
    
    train_ds = StockDataset(train_data, seq_len)
    test_ds = StockDataset(test_data, seq_len)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for x, y in train_loader:
            x, y = x.unsqueeze(-1).to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(train_loader))
        
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.unsqueeze(-1).to(device)
            pred = model(x)
            preds.extend(pred.cpu().numpy().flatten())
            actuals.extend(y.numpy().flatten())
    
    preds = np.array(preds)
    actuals = np.array(actuals)
    
    preds_inv = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    actuals_inv = scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()
    
    dir_actual = np.diff(actuals_inv) > 0
    dir_pred = np.diff(preds_inv) > 0
    dir_acc = np.mean(dir_actual == dir_pred) if len(dir_actual) > 0 else 0
    
    last_seq = torch.FloatTensor(prices_scaled[-seq_len:]).unsqueeze(0).unsqueeze(-1).to(device)
    with torch.no_grad():
        next_pred_scaled = model(last_seq).cpu().numpy()[0, 0]
    next_pred = scaler.inverse_transform([[next_pred_scaled]])[0, 0]
    current = prices[-1]
    change_pct = (next_pred - current) / current * 100
    
    return {
        'current_price': current,
        'next_pred': next_pred,
        'change_pct': change_pct,
        'dir_acc': dir_acc,
        'losses': losses,
        'preds_inv': preds_inv,
        'actuals_inv': actuals_inv,
        'train_size': train_size
    }

# -----------------------
# Single Ticker Handlers
# -----------------------
def run_ensemble(ticker, period="2y"):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df.empty:
            st.error("Geen data gevonden voor deze ticker.")
            return
        
        df = add_technical_indicators(df)
        ens = calc_ensemble(df)
        if not ens:
            st.warning("Niet genoeg data voor training.")
            return
            
        current_price = df['Close'].iloc[-1]
        direction = "📈 OMHOOG" if ens['prob_up'] > 0.55 else ("📉 OMLAAG" if ens['prob_up'] < 0.45 else "➡️ NEUTRAAL")
        
        info = stock.info
        company = info.get('longName', ticker)
        sector = info.get('sector', 'N/A')
        market_cap = info.get('marketCap', 0)
        mcap_str = f"${market_cap:,.0f}" if market_cap else "N/A"
        
        st.markdown(f"### 📊 Ensemble ML Analyse voor **{ticker}** ({company})")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Huidige prijs", f"${current_price:.2f}")
        col2.metric("Sector", sector)
        col3.metric("Market Cap", mcap_str)
        col4.metric("Signaal", direction)
        
        st.markdown("#### Model Performance (test set)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Random Forest", f"{ens['rf_acc']:.1%}")
        c2.metric("Gradient Boosting", f"{ens['gb_acc']:.1%}")
        c3.metric("Ensemble Accuracy", f"{ens['ens_acc']:.1%}")
        c4.metric("Kans op stijging", f"{ens['prob_up']:.1%}")
        
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            vertical_spacing=0.05,
                            row_heights=[0.5, 0.25, 0.25],
                            subplot_titles=(f'{ticker} Prijs', 'RSI (14)', 'Ensemble Probability'))
        
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                     low=df['Low'], close=df['Close'], name='Prijs'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA20', line=dict(color='orange')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA50', line=dict(color='blue')), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        fig.add_trace(go.Scatter(x=ens['X_test'].index, y=ens['ensemble_proba'], name='P(up)', line=dict(color='green')), row=3, col=1)
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", row=3, col=1)
        
        fig.update_layout(height=700, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
        
        avg_imp = ens['feature_imp']
        imp_fig = go.Figure(go.Bar(x=avg_imp.values[::-1], y=avg_imp.index[::-1], orientation='h', marker_color='steelblue'))
        imp_fig.update_layout(title="Top 10 Feature Importance", height=400, xaxis_title="Importance")
        st.plotly_chart(imp_fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Fout bij Ensemble analyse: {str(e)}")

def run_lstm(ticker, period="2y", epochs=12):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df.empty or len(df) < 100:
            st.warning("Niet genoeg data voor LSTM.")
            return
            
        lstm = calc_lstm(df, epochs=epochs)
        if not lstm:
            st.warning("Kon LSTM model niet trainen.")
            return
            
        direction = "📈 OMHOOG" if lstm['change_pct'] > 0.5 else ("📉 OMLAAG" if lstm['change_pct'] < -0.5 else "➡️ NEUTRAAL")
        
        st.markdown(f"### 🧠 LSTM Deep Learning Analyse voor **{ticker}**")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Huidige prijs", f"${lstm['current_price']:.2f}")
        c2.metric("Voorspelde Close", f"${lstm['next_pred']:.2f}", f"{lstm['change_pct']:+.2f}%")
        c3.metric("Signaal", direction)
        c4.metric("Direction Accuracy", f"{lstm['dir_acc']:.1%}")
        
        test_dates = df.index[lstm['train_size']:]
        min_len = min(len(test_dates), len(lstm['actuals_inv']))
        test_dates = test_dates[-min_len:]
        actuals_inv = lstm['actuals_inv'][-min_len:]
        preds_inv = lstm['preds_inv'][-min_len:]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Historische Close', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=test_dates, y=actuals_inv, name='Actual (test)', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=test_dates, y=preds_inv, name='LSTM Prediction', line=dict(color='red', dash='dash')))
        fig.add_trace(go.Scatter(x=[df.index[-1]], y=[lstm['next_pred']], mode='markers+text',
                                 name='Next Pred', marker=dict(size=12, color='orange'),
                                 text=[f"${lstm['next_pred']:.2f}"], textposition="top center"))
        
        fig.update_layout(title=f"LSTM Price Prediction - {ticker}", height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        loss_fig = go.Figure(go.Scatter(y=lstm['losses'], mode='lines+markers', name='Train Loss'))
        loss_fig.update_layout(title="Training Loss per Epoch", height=300)
        st.plotly_chart(loss_fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Fout bij LSTM analyse: {str(e)}")

# -----------------------
# Multi Ticker AI Scan (Ensemble + LSTM)
# -----------------------
def multi_ai_scan(tickers_str, epochs=10):
    tickers = [t.strip().upper() for t in tickers_str.replace(',', ' ').split() if t.strip()]
    if not tickers:
        st.info("Voer minstens één ticker in (bijv. AAPL MSFT TSLA)")
        return
    
    summary_data = []
    progress_bar = st.progress(0)
    
    for idx, t in enumerate(tickers[:6]):
        st.text(f"⚡ Aan het scannen: {t} (Ensemble ML & Deep Learning LSTM)...")
        try:
            stock = yf.Ticker(t)
            df = stock.history(period="2y")
            
            if df.empty or len(df) < 100:
                summary_data.append({
                    "Ticker": t, "Prijs": "N/A", "Ensemble Stijgkans": "N/A",
                    "Ensemble Accuracy": "N/A", "Ensemble Signaal": "Geen data",
                    "LSTM Voorspelling": "N/A", "Verwachte Verandering": "N/A",
                    "LSTM Accuracy": "N/A", "LSTM Signaal": "N/A", "Gecombineerde AI Score": "N/A"
                })
                continue
                
            df = add_technical_indicators(df)
            
            # Run ML
            ens = calc_ensemble(df)
            # Run DL
            lstm = calc_lstm(df, epochs=epochs)
            
            price = df['Close'].iloc[-1]
            
            # Ensemble metrics
            ens_prob = ens['prob_up'] if ens else 0.5
            ens_acc = f"{ens['ens_acc']:.1%}" if ens else "N/A"
            ens_sig = "📈 BULLISH" if ens_prob > 0.55 else ("📉 BEARISH" if ens_prob < 0.45 else "➡️ NEUTRAAL")
            
            # LSTM metrics
            lstm_pred = f"${lstm['next_pred']:.2f}" if lstm else "N/A"
            lstm_pct = lstm['change_pct'] if lstm else 0.0
            lstm_acc = f"{lstm['dir_acc']:.1%}" if lstm else "N/A"
            lstm_sig = "📈 BULLISH" if lstm_pct > 0.5 else ("📉 BEARISH" if lstm_pct < -0.5 else "➡️ NEUTRAAL")
            
            # Total Ensemble + LSTM Combined Rating (1-10)
            score = (ens_prob * 5) + (min(max((lstm_pct + 5) / 10, 0), 1) * 5)
            
            summary_data.append({
                "Ticker": t,
                "Prijs": f"${price:.2f}",
                "Gecombineerde AI Score": f"{score:.1f} / 10",
                "Ensemble Stijgkans": f"{ens_prob:.1%}",
                "Ensemble Signaal": ens_sig,
                "Ensemble Acc": ens_acc,
                "LSTM Voorspelling": lstm_pred,
                "LSTM Verwachte Δ": f"{lstm_pct:+.2f}%",
                "LSTM Signaal": lstm_sig,
                "LSTM Acc": lstm_acc,
            })
            
        except Exception as e:
            summary_data.append({
                "Ticker": t, "Prijs": "Error", "Gecombineerde AI Score": "N/A",
                "Ensemble Stijgkans": "N/A", "Ensemble Signaal": str(e)[:25],
                "Ensemble Acc": "N/A", "LSTM Voorspelling": "N/A",
                "LSTM Verwachte Δ": "N/A", "LSTM Signaal": "N/A", "LSTM Acc": "N/A"
            })
            
        progress_bar.progress((idx + 1) / min(len(tickers), 6))
        
    st.markdown("### 📋 Multi-Ticker AI Scan Overzicht")
    df_res = pd.DataFrame(summary_data)
    
    # Render interactive DataFrame
    st.dataframe(df_res, use_container_width=True)

# -----------------------
# Streamlit Interface
# -----------------------
st.title("🤖 AI Stock Trading Analysis Dashboard")
st.markdown("**Ensemble Machine Learning** (Random Forest + Gradient Boosting) + **Deep Learning LSTM**")

tab1, tab2, tab3 = st.tabs(["🔍 Quick Multi-Ticker Scan", "📈 Single Ticker Ensemble ML", "🧠 Single Ticker Deep Learning LSTM"])

with tab1:
    st.markdown("### ⚡ AI Multi-Ticker Scan (Ensemble ML + Deep Learning LSTM)")
    st.info("Voer meerdere tickers in om zowel het Machine Learning model als het Neural Network tegelijk te laten rekenen.")
    
    col_a, col_b = st.columns([3, 1])
    with col_a:
        multi_in = st.text_area("Tickers (gescheiden door spatie of komma, max 6)", value="AAPL MSFT GOOGL TSLA NVDA", height=70)
    with col_b:
        scan_epochs = st.slider("LSTM Epochs (snelheid vs nauwkeurigheid)", 5, 20, 8)
        
    if st.button("🚀 Start Volledige AI Scan", type="primary"):
        with st.spinner("Modellen worden getraind op historische koersen..."):
            multi_ai_scan(multi_in, epochs=scan_epochs)

with tab2:
    st.markdown("### Ensemble: RandomForest + GradientBoosting op technische indicators + returns")
    col1, col2 = st.columns([2, 1])
    with col1:
        ens_ticker = st.text_input("Ticker", value="AAPL", key="ens_t")
    with col2:
        ens_period = st.selectbox("Periode data", ["1y", "2y", "5y", "max"], index=1, key="ens_p")
    
    if st.button("Analyseer met Ensemble", type="primary"):
        with st.spinner("Modellen trainen en analyseren..."):
            run_ensemble(ens_ticker, ens_period)

with tab3:
    st.markdown("### LSTM voor time-series prijsvoorspelling (sequenties van closes)")
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        lstm_ticker = st.text_input("Ticker", value="AAPL", key="lstm_t")
    with col2:
        lstm_period = st.selectbox("Periode data", ["1y", "2y", "5y"], index=1, key="lstm_p")
    with col3:
        lstm_epochs = st.slider("Training Epochs", 5, 30, 12, key="lstm_e")
    
    if st.button("Train & Analyseer LSTM", type="primary"):
        with st.spinner("Neural network trainen..."):
            run_lstm(lstm_ticker, lstm_period, lstm_epochs)
