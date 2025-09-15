# desktop_app_final_key_fix.py

import tkinter
import TkEasyGUI as eg
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
import threading
import queue

# --- バックエンド処理 (変更なし) ---
warnings.filterwarnings('ignore', category=FutureWarning, module='yfinance.utils')

def analyze_price_convergence(ticker, period="3mo", window_size=5):
    """株価分析のコア関数"""
    stock_data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = stock_data.columns.get_level_values(0)
    if stock_data.empty or len(stock_data) < window_size * 2: return None
    stock_data['SMA5'] = stock_data['Close'].rolling(window=5).mean()
    stock_data.dropna(inplace=True)
    if len(stock_data) < 2: return None
    cv_list = []
    for _, row in stock_data.iterrows():
        values = np.array([row['Open'], row['Close'], row['SMA5']])
        mean, std = np.mean(values), np.std(values)
        cv = std / mean if mean > 0 else 0
        cv_list.append(cv)
    stock_data['Convergence_CV'] = cv_list
    stock_data['Convergence_Score'] = stock_data['Convergence_CV'].rolling(window=window_size).mean()
    stock_data.dropna(inplace=True)
    return stock_data

def analysis_thread_func(thread_queue: queue.Queue, csv_path: str, min_score: float, max_score: float):
    """分析処理をバックグラウンドで実行する関数"""
    try:
        try:
            df = pd.read_csv(csv_path, usecols=[0, 1], encoding='cp932')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, usecols=[0, 1], encoding='utf-8')
        df.columns = ['コード', '銘柄名']
        df['コード'] = df['コード'].astype(str)

        results_list = []
        total = len(df)
        for i, row in df.iterrows():
            progress = (i + 1, total, row['銘柄名'])
            thread_queue.put(("-THREAD-UPDATE-", progress))
            convergence_data = analyze_price_convergence(f"{row['コード']}.T")
            if convergence_data is not None and len(convergence_data) >= 2:
                latest_close = convergence_data['Close'].iloc[-1]
                previous_close = convergence_data['Close'].iloc[-2]
                change = ((latest_close - previous_close) / previous_close) * 100 if previous_close > 0 else 0
                latest_row = convergence_data.iloc[-1]
                results_list.append({
                    'コード': row['コード'], '銘柄名': row['銘柄名'], '現在値': latest_close,
                    '前日比(%)': change, 'Open': latest_row['Open'], 'Close': latest_row['Close'],
                    'SMA5': latest_row['SMA5'], 'Volume': latest_row['Volume'],
                    'Convergence_Score': latest_row['Convergence_Score']
                })
        
        if results_list:
            final_df = pd.DataFrame(results_list)
            sorted_df = final_df.sort_values(by='Convergence_Score', ascending=True)
            filtered_df = sorted_df[
                (sorted_df['Convergence_Score'] >= min_score) &
                (sorted_df['Convergence_Score'] <= max_score)
            ]
        else:
            filtered_df = pd.DataFrame()
        
        thread_queue.put(("-THREAD-DONE-", filtered_df))
    except Exception as e:
        thread_queue.put(("-THREAD-ERROR-", str(e)))

# --- UI (ユーザーインターフェース) 部分 ---
eg.theme("clam")
header = ['コード', '銘柄名', '現在値', '前日比(%)', 'Open', 'Close', 'SMA5', 'Volume', 'Convergence_Score']
layout = [
    [eg.Text("分析したい銘柄リストのCSVファイルを選択してください。")],
    [
        eg.Input(key="-CSV_PATH-", readonly=True, background_color="lightgray"),
        eg.FileBrowse("ファイル選択", file_types=(("CSVファイル", "*.csv"), ("すべてのファイル", "*.*")))
    ],
    [
        eg.Text("Convergence_Score 最小値:"), eg.Input("0.002", key="-MIN_SCORE-", size=(10, 1)),
        eg.Text("最大値:"), eg.Input("0.005", key="-MAX_SCORE-", size=(10, 1))
    ],
    [
        eg.Button("分析を実行", key="-RUN-", button_color=("white", "blue")),
        eg.Button("結果を保存", key="-SAVE-", disabled=True),
        eg.Button("終了", key="-EXIT-", button_color=("white", "red"))
    ],
    [eg.Text("進捗:", size=(10, 1)), eg.Text("", size=(42, 1), key='-PROG-TEXT-')],
    [eg.Text("ステータス: 準備完了", key="-STATUS-", size=(60, 1))],
    [eg.Table(
        values=[], headings=header, key="-TABLE-", auto_size_columns=False,
        justification='right', height=40, col_widths=[5, 15, 8, 8, 8, 8, 8, 10, 15]
    )]
]
window = eg.Window("株価収束度分析デスクトップアプリ", layout, resizable=True)

df_to_save = None
thread_queue = queue.Queue()
while True:
    event, values = window.read(timeout=100)
    if event in (eg.WIN_CLOSED, "-EXIT-"):
        break

    try:
        message_key, message_value = thread_queue.get_nowait()
        if message_key == "-THREAD-UPDATE-":
            current, total, name = message_value
            percent = int((current / total) * 100)
            bar_fill = '■' * (percent // 10)
            bar_empty = '─' * (10 - (percent // 10))
            bar_text = f"[{bar_fill}{bar_empty}] {percent}%"
            window["-PROG-TEXT-"].update(bar_text)
            window["-STATUS-"].update(f"ステータス: 分析中... {name} ({current}/{total})")

        elif message_key == "-THREAD-DONE-":
            df_result = message_value
            df_to_save = df_result.copy()
            window["-PROG-TEXT-"].update(f"[{'■' * 10}] 100%")
            if not df_result.empty:
                df_display = df_result.copy()
                for col in ['現在値', '前日比(%)', 'Open', 'Close', 'SMA5']:
                    df_display[col] = df_display[col].map('{:,.2f}'.format)
                df_display['Volume'] = df_display['Volume'].map('{:,.0f}'.format)
                df_display['Convergence_Score'] = df_display['Convergence_Score'].map('{:.5f}'.format)
                window["-TABLE-"].update(values=df_display.values.tolist())
                window["-SAVE-"].update(disabled=False)
                eg.popup("分析が完了しました。")
            else:
                window["-TABLE-"].update(values=[])
                eg.popup("分析完了。条件に一致する銘柄はありませんでした。")
            window["-STATUS-"].update("ステータス: 準備完了")
            window["-RUN-"].update(disabled=False)

        elif message_key == "-THREAD-ERROR-":
            eg.popup_error(f"分析中にエラーが発生しました。\n{message_value}")
            window["-STATUS-"].update("ステータス: エラーが発生しました")
            window["-RUN-"].update(disabled=False)

    except queue.Empty:
        pass

    if event == "-RUN-":
        df_to_save = None
        csv_path = values["-CSV_PATH-"]
        if not csv_path:
            eg.popup_error("CSVファイルが選択されていません。")
            continue
        try:
            min_score = float(values["-MIN_SCORE-"])
            # --- ▼▼▼ 修正箇所 ▼▼▼ ---
            # ハイフン(-)をアンダースコア(_)に修正
            max_score = float(values["-MAX_SCORE-"])
            # --- ▲▲▲ 修正完了 ▲▲▲ ---
            if min_score > max_score:
                eg.popup_error("最小値が最大値を超えています。")
                continue
        except ValueError:
            eg.popup_error("スコアには数値を入力してください。")
            continue

        window["-RUN-"].update(disabled=True)
        window["-SAVE-"].update(disabled=True)
        window["-STATUS-"].update("ステータス: 分析を開始します...")
        window["-PROG-TEXT-"].update("[          ] 0%")
        
        threading.Thread(
            target=analysis_thread_func,
            args=(thread_queue, csv_path, min_score, max_score),
            daemon=True
        ).start()

    if event == "-SAVE-":
        if df_to_save is None or df_to_save.empty:
            eg.popup_error("保存するデータがありません。")
            continue
        
        save_path = eg.popup_get_file(
            "結果をCSVファイルに保存", save_as=True,
            file_types=(("CSVファイル", "*.csv"),), default_extension=".csv"
        )
        if save_path:
            df_to_save.to_csv(save_path, index=False, encoding="utf-8-sig")
            eg.popup(f"ファイルを保存しました。\n{save_path}")

window.close()