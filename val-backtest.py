# app_upp_streamlit.py
# Streamlit × Backtesting.py で UPP（Uptrend Pullback Pop）戦略を検証
# 使い方:
#   streamlit run app_upp_streamlit.py
# サイドバーでデータソース（CSV or yfinance）と戦略パラメータを設定できます。

import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
from pathlib import Path

# ---- データ取得オプション（yfinanceは任意） ----
try:
    import yfinance as yf
    _HAS_YF = True
except Exception:
    _HAS_YF = False

from backtesting import Backtest, Strategy

# =========================
#     インジケータ関数群
# =========================
def SMA(s, n):
    s = pd.Series(s)
    return s.rolling(int(n)).mean()

def RSI(series, n=2):
    s = pd.Series(series)
    delta = s.diff()
    up = delta.clip(lower=0).rolling(n).mean()
    dn = (-delta.clip(upper=0)).rolling(n).mean()
    # 0除算ガード
    rs = up / (dn.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def BBANDS(close, n=20, k=2):
    c = pd.Series(close)
    m = c.rolling(int(n)).mean()
    sd = c.rolling(int(n)).std(ddof=0)
    upper = m + k * sd
    lower = m - k * sd
    return lower, m, upper

def ATR(high, low, close, n=14):
    h, l, c = map(pd.Series, (high, low, close))
    prev = c.shift(1)
    tr = pd.concat([(h - l), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(int(n)).mean()

def MACD_HIST(close, fast=12, slow=26, sig=9):
    c = pd.Series(close)
    ema = lambda x, n: x.ewm(span=int(n), adjust=False).mean()
    macd = ema(c, fast) - ema(c, slow)
    signal = ema(macd, sig)
    hist = macd - signal
    return macd, signal, hist

# =========================
#     戦略本体（UPP）
# =========================
class UPPStrategy(Strategy):
    # 最適化対応のためパラメータ化
    rsi_th_buy    = 5
    rsi_th_exit   = 70
    bb_n          = 20
    bb_k          = 2.0
    atr_n         = 14
    sma_fast      = 20
    sma_slow      = 200
    atr_min       = 0.008   # 0.8%
    atr_max       = 0.040   # 4.0%
    time_stop     = 10      # 営業日
    atr_init_sl   = 1.5
    atr_trail     = 2.5
    use_hist_slope= True    # MACDヒストの縮小を確認

    def init(self):
        p = self.data
        self.sma200 = self.I(SMA, p.Close, self.sma_slow)
        self.sma20  = self.I(SMA, p.Close, self.sma_fast)
        self.bb_low, self.bb_mid, self.bb_up = self.I(BBANDS, p.Close, self.bb_n, self.bb_k)
        self.rsi2   = self.I(RSI, p.Close, 2)
        self.atr    = self.I(ATR, p.High, p.Low, p.Close, self.atr_n)
        self.macd, self.macds, self.mhist = self.I(MACD_HIST, p.Close)

        self.entry_idx = None
        self.highest   = None

    def next(self):
        i = len(self.data.Close) - 1
        C = float(self.data.Close[-1])
        if not np.isfinite(C):
            return

        atr = float(self.atr[-1]) if np.isfinite(self.atr[-1]) else np.nan
        sma200_now = float(self.sma200[-1]) if np.isfinite(self.sma200[-1]) else np.nan
        sma200_prev = float(self.sma200[-5]) if len(self.sma200) > 5 and np.isfinite(self.sma200[-5]) else np.nan

        # データ不足はスキップ
        if np.isnan([atr, sma200_now, sma200_prev]).any():
            return

        # レジーム: SMA200上 ＆ SMA200が上向き
        regime_ok = (C > sma200_now) and (sma200_now > sma200_prev)

        # ATR割合レンジ
        atrp = atr / C if C else np.nan
        vol_ok = (atrp >= self.atr_min) and (atrp <= self.atr_max)

        # 保有時の管理
        if self.position:
            # 最高値更新
            if (self.highest is None) or (C > self.highest):
                self.highest = C

            # 利確1: BBミドル到達
            if np.isfinite(self.bb_mid[-1]) and (C >= float(self.bb_mid[-1])):
                self.position.close()
                self.entry_idx = None
                self.highest = None
                return

            # 利確2: RSI(2) >= rsi_th_exit
            if np.isfinite(self.rsi2[-1]) and (self.rsi2[-1] >= self.rsi_th_exit):
                self.position.close()
                self.entry_idx = None
                self.highest = None
                return

            # 時間切れ
            if (self.entry_idx is not None) and (i - self.entry_idx >= self.time_stop):
                self.position.close()
                self.entry_idx = None
                self.highest = None
                return

            # トレーリング・ストップ
            if (self.highest is not None) and np.isfinite(atr):
                trail_sl = self.highest - self.atr_trail * atr
                if C <= trail_sl:
                    self.position.close()
                    self.entry_idx = None
                    self.highest = None
                    return

        # 新規エントリー
        if (not self.position) and regime_ok and vol_ok:
            rsi_buy = np.isfinite(self.rsi2[-1]) and (self.rsi2[-1] <= self.rsi_th_buy)
            bb_ok   = np.isfinite(self.bb_low[-1]) and (C <= float(self.bb_low[-1]))

            setup_ok = rsi_buy and bb_ok
            if self.use_hist_slope:
                # MACDヒスト縮小（下落モメが弱まり始めた）: hist[t] > hist[t-1]
                if len(self.mhist) >= 2 and np.isfinite(self.mhist[-1]) and np.isfinite(self.mhist[-2]):
                    setup_ok = setup_ok and (float(self.mhist[-1]) > float(self.mhist[-2]))
                else:
                    setup_ok = False

            if setup_ok:
                self.buy()
                self.entry_idx = i
                self.highest = C

# =========================
#     Streamlit UI
# =========================
st.set_page_config(page_title="UPP Strategy Backtest", layout="wide")
st.title("UPP（Uptrend Pullback Pop）戦略 – Backtesting.py 検証アプリ")

with st.sidebar:
    st.header("データソース")
    src = st.radio("選択", ["CSVアップロード", "yfinance（ティッカー）"])

    if src == "CSVアップロード":
        file = st.file_uploader("CSVをアップロード（列: Date,Open,High,Low,Close,Volume）", type=["csv"])
    else:
        if not _HAS_YF:
            st.warning("yfinance が見つかりません。`pip install yfinance` を実行してください。")
        ticker = st.text_input("ティッカー（例: ^N225, 7203.T, SPY, AAPL）", value="AAPL")
        start  = st.date_input("開始日", value=dt.date.today() - dt.timedelta(days=365*5))
        end    = st.date_input("終了日", value=dt.date.today())

    st.header("戦略パラメータ")
    rsi_th_buy    = st.slider("RSI(2) 買い閾値", 1, 20, 5)
    rsi_th_exit   = st.slider("RSI(2) 利確閾値", 50, 95, 70)
    bb_n          = st.slider("BB 期間 (n)", 10, 40, 20)
    bb_k          = st.slider("BB σ (k)", 1.0, 3.0, 2.0, 0.1)
    atr_n         = st.slider("ATR 期間", 5, 30, 14)
    sma_fast      = st.slider("SMA（ミドル）", 10, 50, 20)
    sma_slow      = st.slider("SMA（長期/環境）", 150, 300, 200)
    atr_min       = st.slider("ATR% 最小", 0.001, 0.05, 0.008, 0.001, format="%.3f")
    atr_max       = st.slider("ATR% 最大", 0.01, 0.10, 0.040, 0.001, format="%.3f")
    time_stop     = st.slider("時間切れ（日）", 3, 20, 10)
    atr_init_sl   = st.slider("初期SL: ATR倍率", 0.5, 3.0, 1.5, 0.1)
    atr_trail     = st.slider("トレールSL: ATR倍率", 1.0, 4.0, 2.5, 0.1)
    use_hist_slope= st.checkbox("MACDヒスト縮小を確認（ダマシ減）", value=True)

    st.header("バックテスト設定")
    cash        = st.number_input("初期資金", value=100_000, step=10_000)
    commission  = st.number_input("手数料（率）", value=0.0005, step=0.0001, format="%.4f")
    run_btn     = st.button("バックテスト実行")

# =========================
#     データ読み込み
# =========================
@st.cache_data(show_spinner=False)
def load_from_yf(ticker: str, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
    if df.empty:
        return df
    # yfinance の MultiIndex 対策
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.reset_index()
    df.rename(columns={"Date":"Date"}, inplace=True)
    # 必要列の整形
    need = ["Date","Open","High","Low","Close","Volume"]
    df = df[need]
    return df

def sanitize_prices(df: pd.DataFrame) -> pd.DataFrame:
    # 列名の正規化
    cols_lower = {c:c for c in df.columns}
    rename_map = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl == "date": rename_map[c] = "Date"
        elif cl == "open": rename_map[c] = "Open"
        elif cl == "high": rename_map[c] = "High"
        elif cl == "low": rename_map[c] = "Low"
        elif cl in ("close","adj close","adj_close","adjusted_close"): rename_map[c] = "Close"
        elif cl == "volume": rename_map[c] = "Volume"
    df = df.rename(columns=rename_map)

    # 型変換
    df["Date"] = pd.to_datetime(df["Date"])
    for c in ["Open","High","Low","Close","Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Open","High","Low","Close"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df

def get_prices():
    if src == "CSVアップロード":
        if file is None:
            return pd.DataFrame()
        df = pd.read_csv(file)
        return sanitize_prices(df)
    else:
        if not _HAS_YF:
            return pd.DataFrame()
        df = load_from_yf(ticker, start, end)
        if df.empty:
            return df
        return sanitize_prices(df)

# =========================
#     実行
# =========================
df_prices = get_prices()

if run_btn:
    if df_prices.empty:
        st.error("データが読み込めていません。CSVをアップロードするか、yfinanceで範囲を見直してください。")
        st.stop()

    # SMA200を使うので最低でも200本以上必要（バッファ含め300本以上推奨）
    if len(df_prices) < max(sma_slow + 10, 250):
        st.warning(f"データ本数が少ない可能性があります（{len(df_prices)} 本）。SMA{str(sma_slow)}の計算には十分な本数が必要です。")
    st.write(f"読み込みデータ: {df_prices['Date'].min().date()} 〜 {df_prices['Date'].max().date()}（{len(df_prices)} 本）")

    # Strategyパラメータを注入
    class _UPP(UPPStrategy):
        rsi_th_buy     = int(rsi_th_buy)
        rsi_th_exit    = int(rsi_th_exit)
        bb_n           = int(bb_n)
        bb_k           = float(bb_k)
        atr_n          = int(atr_n)
        sma_fast       = int(sma_fast)
        sma_slow       = int(sma_slow)
        atr_min        = float(atr_min)
        atr_max        = float(atr_max)
        time_stop      = int(time_stop)
        atr_init_sl    = float(atr_init_sl)
        atr_trail      = float(atr_trail)
        use_hist_slope = bool(use_hist_slope)

    df_bt = df_prices.set_index("Date")[["Open","High","Low","Close","Volume"]].copy()

    bt = Backtest(
        df_bt,
        _UPP,
        cash=float(cash),
        commission=float(commission),
        exclusive_orders=True
    )
    stats = bt.run()

    # 結果表示
    st.subheader("成績サマリー")
    # 見やすい主要項目だけを並べる（存在チェック付）
    keys_pref = [
        'Start', 'End', 'Duration', 'Exposure Time [%]',
        'Equity Final [$]', 'Return [%]', 'Buy & Hold Return [%]',
        '# Trades', 'Win Rate [%]', 'Best Trade [%]', 'Worst Trade [%]',
        'Avg. Trade [%]', 'Max. Drawdown [%]', 'Sharpe Ratio', 'SQN'
    ]
    s = stats.copy()
    s.index = s.index.astype(str)
    summary = {k: s[k] for k in keys_pref if k in s.index}
    st.dataframe(pd.DataFrame.from_dict(summary, orient="index"), use_container_width=True)

    # 取引一覧（あれば）
    trades = None
    if '_trades' in stats:
        trades = stats['_trades']
    elif hasattr(stats, 'trades'):
        trades = stats.trades
    if trades is not None and len(trades):
        st.subheader("トレード一覧")
        st.dataframe(trades, use_container_width=True, height=400)

    # チャート（Bokeh HTML を埋め込み）
    st.subheader("バックテスト チャート")
    report_path = Path("bt_report.html")
    bt.plot(filename=str(report_path), open_browser=False)
    html = report_path.read_text(encoding="utf-8")
    st.components.v1.html(html, height=900, scrolling=True)

else:
    st.info("左のサイドバーでデータとパラメータを設定し、「バックテスト実行」を押してください。")
