#!/usr/bin/env python3
"""
AI Stock Trading Analysis Dashboard
Ensemble ML (RandomForest + GradientBoosting) + Deep Learning (LSTM)
User can input tickers for analysis.
"""

import gradio as gr
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
# Ensemble ML Section
# -----------------------
def run_ensemble(ticker, period="2y"):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df.empty:
            return "Geen data gevonden voor deze ticker.", None, None, None
        
        df = add_technical_indicators(df)
        
        feature_cols = [
            'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'MACD', 'MACD_signal', 'ADX',
            'RSI', 'Stoch', 'CCI', 'BB_high', 'BB_low', 'ATR', 'OBV', 'MFI', 'Return'
        ]
        available = [c for c in feature_cols if c in df.columns]
        X = df[available]
        y = df['Target']
        
        if len(X) < 100:
            return "Niet genoeg data voor training (minimaal ~100 rijen na indicators).", None, None, None
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Random Forest
        rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_acc = accuracy_score(y_test, rf_pred)
        
        # Gradient Boosting
        gb = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
        gb.fit(X_train, y_train)
        gb_pred = gb.predict(X_test)
        gb_acc = accuracy_score(y_test, gb_pred)
        
        # Ensemble vote
        rf_proba = rf.predict_proba(X_test)[:, 1]
        gb_proba = gb.predict_proba(X_test)[:, 1]
        ensemble_proba = (rf_proba + gb_proba) / 2
        ensemble_pred = (ensemble_proba > 0.5).astype(int)
        ens_acc = accuracy_score(y_test, ensemble_pred)
        
        # Latest prediction
        latest_features = X.iloc[[-1]]
        rf_latest = rf.predict_proba(latest_features)[0, 1]
        gb_latest = gb.predict_proba(latest_features)[0, 1]
        ens_latest = (rf_latest + gb_latest) / 2
        direction = "📈 OMHOOG" if ens_latest > 0.55 else ("📉 OMLAAG" if ens_latest < 0.45 else "➡️ NEUTRAAL")
        
        # Feature importance
        rf_imp = pd.Series(rf.feature_importances_, index=available)
        gb_imp = pd.Series(gb.feature_importances_, index=available)
        avg_imp = ((rf_imp + gb_imp) / 2).sort_values(ascending=False).head(10)
        
        # Info
        info = stock.info
        company = info.get('longName', ticker)
        sector = info.get('sector', 'N/A')
        market_cap = info.get('marketCap', 0)
        current_price = df['Close'].iloc[-1]
        mcap_str = f"${market_cap:,.0f}" if market_cap else "N/A"
        
        summary = f"""
### 📊 Ensemble ML Analyse voor **{ticker}** ({company})

**Huidige prijs:** ${current_price:.2f}  
**Sector:** {sector}  
**Market Cap:** {mcap_str}

#### Model Performance (test set)
- **Random Forest Accuracy:** {rf_acc:.1%}
- **Gradient Boosting Accuracy:** {gb_acc:.1%}
- **Ensemble Accuracy:** {ens_acc:.1%}

#### Laatste voorspelling (volgende dag)
- **Ensemble kans op stijging:** {ens_latest:.1%}
- **Signaal:** {direction}

#### Top Feature Importance
"""
        for feat, imp in avg_imp.items():
            summary += f"- {feat}: {imp:.3f}\n"
        
        # Charts
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
        
        test_idx = X_test.index
        fig.add_trace(go.Scatter(x=test_idx, y=ensemble_proba, name='P(up)', line=dict(color='green')), row=3, col=1)
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", row=3, col=1)
        
        fig.update_layout(height=800, xaxis_rangeslider_visible=False, title_text=f"Ensemble ML Dashboard - {ticker}")
        
        imp_fig = go.Figure(go.Bar(x=avg_imp.values[::-1], y=avg_imp.index[::-1], orientation='h', marker_color='steelblue'))
        imp_fig.update_layout(title="Top 10 Feature Importance", height=400, xaxis_title="Importance")
        
        return summary, fig, imp_fig, f"Ensemble klaar voor {ticker}"
    
    except Exception as e:
        return f"Fout bij Ensemble analyse: {str(e)}", None, None, "Error"

# -----------------------
# Deep Learning LSTM Section
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

def run_lstm(ticker, period="2y", epochs=15, seq_len=30):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df.empty or len(df) < 100:
            return "Niet genoeg data voor LSTM.", None, None
        
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
        direction = "📈 OMHOOG" if change_pct > 0.5 else ("📉 OMLAAG" if change_pct < -0.5 else "➡️ NEUTRAAL")
        
        summary = f"""
### 🧠 LSTM Deep Learning Analyse voor **{ticker}**

**Huidige prijs:** ${current:.2f}  
**Voorspelde volgende close:** ${next_pred:.2f} ({change_pct:+.2f}%)  
**Signaal:** {direction}

#### Model Metrics (test set)
- **Direction Accuracy:** {dir_acc:.1%}
- **MAE:** ${np.mean(np.abs(preds_inv - actuals_inv)):.2f}
- **MSE:** {np.mean((preds_inv - actuals_inv)**2):.2f}
"""
        
        test_dates = df.index[train_size:]
        min_len = min(len(test_dates), len(actuals_inv))
        test_dates = test_dates[-min_len:]
        actuals_inv = actuals_inv[-min_len:]
        preds_inv = preds_inv[-min_len:]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Historische Close', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=test_dates, y=actuals_inv, name='Actual (test)', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=test_dates, y=preds_inv, name='LSTM Prediction', line=dict(color='red', dash='dash')))
        fig.add_trace(go.Scatter(x=[df.index[-1]], y=[next_pred], mode='markers+text',
                                 name='Next Pred', marker=dict(size=12, color='orange'),
                                 text=[f"${next_pred:.2f}"], textposition="top center"))
        
        fig.update_layout(title=f"LSTM Price Prediction - {ticker}", height=500)
        
        loss_fig = go.Figure(go.Scatter(y=losses, mode='lines+markers', name='Train Loss'))
        loss_fig.update_layout(title="Training Loss per Epoch", height=300)
        
        return summary, fig, loss_fig
    
    except Exception as e:
        return f"Fout bij LSTM analyse: {str(e)}", None, None

# -----------------------
# Multi Ticker Scan
# -----------------------
def quick_scan(tickers_str):
    tickers = [t.strip().upper() for t in tickers_str.replace(',', ' ').split() if t.strip()]
    if not tickers:
        return "Voer minstens één ticker in (bijv. AAPL MSFT TSLA)"
    
    results = []
    for t in tickers[:8]:
        try:
            stock = yf.Ticker(t)
            df = stock.history(period="6mo")
            if df.empty:
                results.append(f"**{t}**: Geen data")
                continue
            df = add_technical_indicators(df)
            last = df.iloc[-1]
            rsi = last['RSI']
            macd = last['MACD']
            signal = last['MACD_signal']
            price = last['Close']
            sma20 = last['SMA_20']
            
            trend = "Bullish" if price > sma20 and macd > signal else ("Bearish" if price < sma20 and macd < signal else "Mixed")
            rsi_status = "Overbought" if rsi > 70 else ("Oversold" if rsi < 30 else "Neutral")
            
            results.append(f"**{t}** | Prijs: ${price:.2f} | RSI: {rsi:.1f} ({rsi_status}) | Trend: {trend}")
        except Exception as e:
            results.append(f"**{t}**: Error - {str(e)[:50]}")
    
    return "\n\n".join(results)

# -----------------------
# Gradio Interface (ALL UI COMPONENT CALLS MUST BE INSIDE THIS BLOCK)
# -----------------------
with gr.Blocks(title="AI Stock Trading Dashboard", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🤖 AI Stock Trading Analysis Dashboard
    **Ensemble Machine Learning** (Random Forest + Gradient Boosting) + **Deep Learning LSTM**
    """)
    
    with gr.Tab("📈 Ensemble ML (Tabular)"):
        gr.Markdown("### Ensemble: RandomForest + GradientBoosting op technische indicators + returns")
        with gr.Row():
            ens_ticker = gr.Textbox(label="Ticker", value="AAPL", placeholder="Bijv. AAPL")
            ens_period = gr.Dropdown(["1y", "2y", "5y", "max"], value="2y", label="Periode data")
            ens_btn = gr.Button("Analyseer met Ensemble", variant="primary")
        
        ens_out = gr.Markdown()
        with gr.Row():
            ens_chart = gr.Plot(label="Prijs + RSI + Probability")
            ens_imp = gr.Plot(label="Feature Importance")
        ens_status = gr.Textbox(label="Status", interactive=False)
        
        ens_btn.click(run_ensemble, inputs=[ens_ticker, ens_period], outputs=[ens_out, ens_chart, ens_imp, ens_status])
    
    with gr.Tab("🧠 Deep Learning LSTM"):
        gr.Markdown("### LSTM voor time-series prijsvoorspelling (sequenties van closes)")
        with gr.Row():
            lstm_ticker = gr.Textbox(label="Ticker", value="AAPL", placeholder="Bijv. AAPL")
            lstm_period = gr.Dropdown(["1y", "2y", "5y"], value="2y", label="Periode data")
            lstm_epochs = gr.Slider(5, 30, value=12, step=1, label="Training Epochs")
            lstm_btn = gr.Button("Train & Analyseer LSTM", variant="primary")
        
        lstm_out = gr.Markdown()
        with gr.Row():
            lstm_chart = gr.Plot(label="Prijs + Predictions")
            lstm_loss = gr.Plot(label="Training Loss")
        
        lstm_btn.click(run_lstm, inputs=[lstm_ticker, lstm_period, lstm_epochs], outputs=[lstm_out, lstm_chart, lstm_loss])
    
    with gr.Tab("🔍 Quick Multi-Ticker Scan"):
        gr.Markdown("### Snelle technische scan voor meerdere tickers")
        multi_in = gr.Textbox(label="Tickers (spatie of komma gescheiden)", value="AAPL MSFT GOOGL TSLA NVDA", lines=2)
        multi_btn = gr.Button("Scan", variant="primary")
        multi_out = gr.Markdown()
        
        multi_btn.click(quick_scan, inputs=multi_in, outputs=multi_out)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
