import TkEasyGUI as eg
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import japanize_matplotlib  # 日本語文字化け対応
from datetime import datetime, timedelta
import warnings
import io
warnings.filterwarnings('ignore')

# matplotlib設定
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 100

def calculate_convergence_metrics(df, window=10):
    """
    株価の収束度を計算する関数
    """
    # 5日移動平均線を計算
    df['MA5'] = df['Close'].rolling(window=5).mean()
    
    # 1. 変動係数（Coefficient of Variation）
    rolling_mean = df['Close'].rolling(window=window).mean()
    rolling_std = df['Close'].rolling(window=window).std()
    df['CV'] = (rolling_std / rolling_mean) * 100
    
    # 2. レンジ比率（高値-安値を終値で正規化）
    df['Range'] = df['High'] - df['Low']
    df['Range_Ratio'] = (df['Range'] / df['Close']) * 100
    df['Range_Ratio_MA'] = df['Range_Ratio'].rolling(window=window).mean()
    
    # 3. ATR（Average True Range）を終値で正規化
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=window).mean()
    df['ATR_Ratio'] = (df['ATR'] / df['Close']) * 100
    
    # 4. ボリンジャーバンド幅
    df['BB_Upper'] = rolling_mean + (rolling_std * 2)
    df['BB_Lower'] = rolling_mean - (rolling_std * 2)
    df['BB_Width'] = ((df['BB_Upper'] - df['BB_Lower']) / rolling_mean) * 100
    
    # 5. 価格変動の標準偏差（対数リターン）
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Volatility'] = df['Log_Return'].rolling(window=window).std() * 100
    
    # 6. 収束スコア（複合指標）
    df['Convergence_Score'] = (
        (1 / (1 + df['CV'].fillna(100))) * 25 +
        (1 / (1 + df['Range_Ratio_MA'].fillna(100))) * 25 +
        (1 / (1 + df['ATR_Ratio'].fillna(100))) * 25 +
        (1 / (1 + df['Volatility'].fillna(100))) * 25
    )
    
    return df

def detect_convergence_periods(df, cv_threshold=2.0, range_threshold=2.0, min_days=5):
    """
    連続した収束期間を検出
    """
    # 収束フラグ（閾値以下を収束と判定）
    df['Is_Converged'] = (
        (df['CV'] < cv_threshold) & 
        (df['Range_Ratio_MA'] < range_threshold)
    ).astype(int)
    
    convergence_periods = []
    
    # 収束フラグが1の連続期間を検出
    df['Group'] = (df['Is_Converged'] != df['Is_Converged'].shift()).cumsum()
    
    for group_id in df[df['Is_Converged'] == 1]['Group'].unique():
        period_df = df[(df['Group'] == group_id) & (df['Is_Converged'] == 1)]
        
        if len(period_df) >= min_days:
            start_date = period_df.index[0]
            end_date = period_df.index[-1]
            duration = len(period_df)
            avg_cv = period_df['CV'].mean()
            
            convergence_periods.append({
                'start': start_date.strftime('%Y-%m-%d'),
                'end': end_date.strftime('%Y-%m-%d'),
                'duration': duration,
                'avg_cv': round(avg_cv, 2)
            })
    
    return convergence_periods

def create_convergence_plot(df, ticker, convergence_periods):
    """
    収束分析の可視化
    """
    fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
    
    # 1. 株価とMA5
    ax1 = axes[0]
    ax1.plot(df.index, df['Close'], label='終値', color='black', linewidth=1)
    ax1.plot(df.index, df['MA5'], label='5日移動平均', color='blue', alpha=0.7)
    
    # 収束期間をハイライト
    for period in convergence_periods:
        start = pd.to_datetime(period['start'])
        end = pd.to_datetime(period['end'])
        ax1.axvspan(start, end, alpha=0.2, color='red', 
                   label='収束期間' if period == convergence_periods[0] else "")
    
    ax1.set_ylabel('株価')
    ax1.set_title(f'{ticker} - 株価収束分析')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. 変動係数（CV）
    ax2 = axes[1]
    ax2.plot(df.index, df['CV'], label='変動係数 (CV)', color='red', linewidth=1)
    ax2.axhline(y=2.0, color='gray', linestyle='--', alpha=0.5, label='閾値 (2%)')
    ax2.set_ylabel('CV (%)')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. レンジ比率とATR比率
    ax3 = axes[2]
    ax3.plot(df.index, df['Range_Ratio_MA'], label='レンジ比率 (移動平均)', 
             color='green', linewidth=1)
    ax3.plot(df.index, df['ATR_Ratio'], label='ATR比率', 
             color='orange', linewidth=1, alpha=0.7)
    ax3.axhline(y=2.0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_ylabel('比率 (%)')
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. 収束スコア
    ax4 = axes[3]
    ax4.fill_between(df.index, 0, df['Convergence_Score'] * 100, 
                     alpha=0.5, color='purple', label='収束スコア')
    ax4.set_ylabel('収束スコア')
    ax4.set_xlabel('日付')
    ax4.legend(loc='best', fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # X軸の日付フォーマット
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        # 月ごとのメジャーティック
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        # 週ごとのマイナーティック
        ax.xaxis.set_minor_locator(mdates.WeekdayLocator(interval=1))
        # フォントサイズを小さくして、より多くの情報を表示
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.tick_params(axis='x', which='minor', length=3)
    
    # X軸ラベルの回転と位置調整
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # レイアウト調整 - 4つのサブプロット全体が確実に表示されるように
    plt.tight_layout(pad=1.5, h_pad=0.8, w_pad=0.3)
    plt.subplots_adjust(bottom=0.10, right=0.96, top=0.94, left=0.08)
    
    return fig

def analyze_stock(ticker, period, window, cv_threshold, range_threshold, min_days):
    """
    株価分析のメイン処理
    """
    try:
        # データ取得
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval='1d')
        
        if df.empty:
            return None, None, "データを取得できませんでした"
        
        # 収束指標を計算
        df = calculate_convergence_metrics(df, window=window)
        
        # 収束期間を検出
        convergence_periods = detect_convergence_periods(
            df, cv_threshold, range_threshold, min_days
        )
        
        return df, convergence_periods, None
        
    except Exception as e:
        return None, None, str(e)

# ========== GUI定義 ==========

def create_placeholder_image():
    """
    初期表示用のプレースホルダー画像を作成
    """
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.text(0.5, 0.5, '株価データを取得してください\n\nティッカーコードを入力し、\n「収束検出実行」ボタンをクリック', 
            ha='center', va='center', fontsize=16, color='gray')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=90, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf.read()

def create_main_layout():
    """
    メインウィンドウのレイアウトを作成
    """
    # タブ定義
    tab1 = [
        [eg.Text("収束分析グラフ", font=("Arial", 12, "bold"))],
        [eg.Image(key="-CANVAS-", size=(950, 700))]
    ]
    
    tab2 = [
        [eg.Text("検出された収束期間", font=("Arial", 12, "bold"))],
        [eg.Multiline(key="-PERIODS-", size=(80, 15), readonly=True, font=("Courier", 10))],
        [eg.Text("", key="-PERIOD_COUNT-", font=("Arial", 11, "bold"))],
        [eg.Button("期間詳細をCSVで保存", key="-EXPORT_PERIODS-", disabled=True)]
    ]
    
    tab3 = [
        [eg.Text("データセット（直近20日）", font=("Arial", 12, "bold"))],
        [eg.Multiline(key="-TABLE-", size=(100, 25), readonly=True, font=("Courier", 9))],
        [eg.Text("※ ○は収束期間、×は非収束期間を示します", font=("Arial", 9), text_color="gray")]
    ]
    
    tab4 = [
        [eg.Text("統計サマリー", font=("Arial", 12, "bold"))],
        [eg.Multiline(key="-SUMMARY-", size=(80, 25), readonly=True, font=("Courier", 10))],
        [eg.Button("サマリーをテキストで保存", key="-EXPORT_SUMMARY-", disabled=True)]
    ]
    
    # メインレイアウト
    layout = [
        # ヘッダー
        [eg.Text("📊 株価収束検出システム", font=("Arial", 16, "bold"), text_color="navy")],
        [eg.HSeparator()],
        
        # 入力部
        [
            eg.Text("ティッカーコード:", size=(15, 1)),
            eg.Input("5247.T", key="-TICKER-", size=(15, 1)),
            eg.Text("期間:", size=(5, 1)),
            eg.Combo(["1mo", "3mo", "6mo", "1y", "2y"], default_value="6mo", key="-PERIOD-", size=(10, 1)),
            eg.Button("🔍 収束検出実行", key="-DETECT-", button_color=("white", "green")),
            eg.Button("🧪 テストデータ", key="-TEST-", button_color=("white", "orange")),
            eg.Button("💾 CSVエクスポート", key="-EXPORT-", disabled=True),
            eg.Button("終了", key="-EXIT-", button_color=("white", "red"))
        ],
        
        # パラメータ設定
        [
            eg.Frame("パラメータ設定", [
                [
                    eg.Text("計算ウィンドウ:", size=(15, 1)),
                    eg.Input("10", key="-WINDOW-", size=(10, 1)),
                    eg.Text("日 (5-20)")
                ],
                [
                    eg.Text("CV閾値(%):", size=(15, 1)),
                    eg.Input("2.0", key="-CV_THRESH-", size=(10, 1)),
                    eg.Text("% (1.0-5.0)")
                ],
                [
                    eg.Text("レンジ閾値(%):", size=(15, 1)),
                    eg.Input("2.0", key="-RANGE_THRESH-", size=(10, 1)),
                    eg.Text("% (1.0-5.0)")
                ],
                [
                    eg.Text("最小連続日数:", size=(15, 1)),
                    eg.Input("5", key="-MIN_DAYS-", size=(10, 1)),
                    eg.Text("日 (3-10)")
                ]
            ])
        ],
        
        # ステータス表示
        [eg.Text("", key="-STATUS-", size=(80, 1), text_color="blue")],
        
        # メトリクス表示
        [
            eg.Frame("分析結果", [
                [
                    eg.Text("分析期間:", size=(10, 1)),
                    eg.Text("---", key="-METRIC1-", size=(10, 1)),
                    eg.Text("収束日数:", size=(10, 1)),
                    eg.Text("---", key="-METRIC2-", size=(10, 1)),
                    eg.Text("収束率:", size=(10, 1)),
                    eg.Text("---", key="-METRIC3-", size=(10, 1)),
                    eg.Text("検出期間数:", size=(12, 1)),
                    eg.Text("---", key="-METRIC4-", size=(10, 1))
                ]
            ])
        ],
        
        # タブグループ
        [eg.TabGroup([
            [eg.Tab("📈 グラフ", tab1)],
            [eg.Tab("📊 収束期間", tab2)],
            [eg.Tab("🔢 データセット", tab3)],
            [eg.Tab("📋 統計サマリー", tab4)]
        ], key="-TABGROUP-")]
    ]
    
    return layout

def main():
    """
    メインアプリケーション
    """
    # ウィンドウ作成
    window = eg.Window("株価収束検出システム", create_main_layout(), 
                      finalize=True, resizable=True, size=(950, 1000))
    
    # 初期画像を設定
    window["-CANVAS-"].update(data=create_placeholder_image())
    
    # データ保持用変数
    current_df = None
    current_ticker = None
    current_periods = None
    current_summary = None
    
    # イベントループ
    while True:
        event, values = window.read(timeout=100)
        
        # 終了処理
        if event in (eg.WIN_CLOSED, "-EXIT-"):
            break
        
        # 収束検出実行
        if event == "-DETECT-":
            ticker = values["-TICKER-"]
            if not ticker:
                eg.popup_error("ティッカーコードを入力してください")
                continue
            
            # ステータス更新
            window["-STATUS-"].update("データを取得中...")
            window.refresh()
            
            # パラメータ取得と検証
            try:
                window_size = int(values["-WINDOW-"])
                cv_thresh = float(values["-CV_THRESH-"])
                range_thresh = float(values["-RANGE_THRESH-"])
                min_days = int(values["-MIN_DAYS-"])
                
                # 範囲チェック
                if not (5 <= window_size <= 20):
                    raise ValueError("計算ウィンドウは5-20の範囲で指定してください")
                if not (1.0 <= cv_thresh <= 5.0):
                    raise ValueError("CV閾値は1.0-5.0の範囲で指定してください")
                if not (1.0 <= range_thresh <= 5.0):
                    raise ValueError("レンジ閾値は1.0-5.0の範囲で指定してください")
                if not (3 <= min_days <= 10):
                    raise ValueError("最小連続日数は3-10の範囲で指定してください")
                    
            except ValueError as e:
                window["-STATUS-"].update(f"エラー: {str(e)}")
                eg.popup_error(f"パラメータエラー:\n{str(e)}")
                continue
            
            # 分析実行
            df, convergence_periods, error = analyze_stock(
                ticker,
                values["-PERIOD-"],
                window_size,
                cv_thresh,
                range_thresh,
                min_days
            )
            
            if error:
                window["-STATUS-"].update(f"エラー: {error}")
                eg.popup_error(f"エラーが発生しました:\n{error}")
                continue
            
            # 統計サマリーを作成
            summary_text = "=" * 70 + "\n"
            summary_text += f"     株価収束分析レポート - {ticker}\n"
            summary_text += "=" * 70 + "\n\n"
            
            summary_text += f"""【基本統計量】
  変動係数（CV）
    平均:     {df['CV'].mean():6.2f}%
    最小:     {df['CV'].min():6.2f}%
    最大:     {df['CV'].max():6.2f}%
    中央値:   {df['CV'].median():6.2f}%
    標準偏差: {df['CV'].std():6.2f}%

【収束統計】
  総収束日数:     {df['Is_Converged'].sum():4d}日
  収束率:         {df['Is_Converged'].sum() / len(df) * 100:6.1f}%
  検出期間数:     {len(convergence_periods):4d}期間
  最長収束期間:   {max([p['duration'] for p in convergence_periods], default=0):4d}日
  平均収束期間:   {np.mean([p['duration'] for p in convergence_periods]) if convergence_periods else 0:6.1f}日
  平均収束スコア: {df['Convergence_Score'].mean() * 100:6.1f}

【分析情報】
  銘柄コード: {ticker}
  分析期間:   {values['-PERIOD-']}
  データ数:   {len(df)}日分
  開始日:     {df.index[0].strftime('%Y-%m-%d')}
  終了日:     {df.index[-1].strftime('%Y-%m-%d')}

【パラメータ設定】
  計算ウィンドウ: {window_size}日
  CV閾値:        {cv_thresh}%
  レンジ閾値:    {range_thresh}%
  最小連続日数:  {min_days}日

【月別収束率】
"""
            # 月別収束率を追加
            monthly_convergence = df.groupby(pd.Grouper(freq='M'))['Is_Converged'].agg(['sum', 'count'])
            monthly_convergence['rate'] = (monthly_convergence['sum'] / monthly_convergence['count'] * 100).round(1)
            
            for month, row in monthly_convergence.iterrows():
                if row['count'] > 0:
                    summary_text += f"  {month.strftime('%Y年%m月')}: "
                    summary_text += f"{row['sum']:3.0f}日/{row['count']:3.0f}日 "
                    summary_text += f"({row['rate']:5.1f}%)"
                    
                    # 収束率をバーグラフで表現
                    bar_length = int(row['rate'] / 5)
                    summary_text += " " + "■ " * bar_length + "\n"
            
            summary_text += "\n" + "=" * 70 + "\n"
            summary_text += f"分析実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            
            # データ保存
            current_df = df
            current_ticker = ticker
            current_periods = convergence_periods
            current_summary = summary_text
            
            # メトリクス更新
            window["-METRIC1-"].update(f"{len(df)}日")
            window["-METRIC2-"].update(f"{df['Is_Converged'].sum()}日")
            window["-METRIC3-"].update(f"{df['Is_Converged'].sum() / len(df) * 100:.1f}%")
            window["-METRIC4-"].update(f"{len(convergence_periods)}件")
            
            # グラフ更新
            fig = create_convergence_plot(df, ticker, convergence_periods)
            
            # 図を一時的にバイト列として保存
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=85, bbox_inches='tight', 
                       facecolor='white', edgecolor='none', pad_inches=0.1)
            buf.seek(0)
            
            # Imageウィジェットに表示
            window["-CANVAS-"].update(data=buf.read())
            plt.close(fig)  # メモリリーク防止
            
            # 収束期間表示
            print("収束期間の表示処理を開始...")
            if convergence_periods:
                periods_text = "=" * 70 + "\n"
                periods_text += "期間\t開始日\t\t終了日\t\t日数\t平均CV(%)\n"
                periods_text += "=" * 70 + "\n"
                
                total_days = 0
                for i, period in enumerate(convergence_periods, 1):
                    periods_text += f" {i:2d}\t{period['start']}\t{period['end']}\t"
                    periods_text += f"{period['duration']:3d}日\t{period['avg_cv']:6.2f}%\n"
                    total_days += period['duration']
                
                periods_text += "-" * 70 + "\n"
                periods_text += f"合計: {len(convergence_periods)}期間、{total_days}日間の収束\n"
                
                print(f"収束期間テキスト作成完了: {len(periods_text)}文字")
                window["-PERIODS-"].update(periods_text)
                window["-PERIOD_COUNT-"].update(
                    f"✓ 合計 {len(convergence_periods)} 期間が検出されました（総収束日数: {total_days}日）"
                )
                print("収束期間タブ更新完了")
            else:
                periods_text = "指定された条件で収束期間は検出されませんでした\n\n" + \
                              "パラメータを調整して再度実行してください:\n" + \
                              "・CV閾値を上げる\n" + \
                              "・レンジ閾値を上げる\n" + \
                              "・最小連続日数を下げる"
                window["-PERIODS-"].update(periods_text)
                window["-PERIOD_COUNT-"].update("収束期間なし")
                print("収束期間なしメッセージ表示完了")
            
            # データテーブル更新
            print("データテーブルの表示処理を開始...")
            table_text = "=" * 85 + "\n"
            table_text += "日付\t\t終値\tCV(%)\tレンジ(%)\tATR(%)\tスコア\t収束\n"
            table_text += "=" * 85 + "\n"
            
            display_df = df[['Close', 'CV', 'Range_Ratio_MA', 'ATR_Ratio', 
                            'Convergence_Score', 'Is_Converged']].tail(20)
            
            for date, row in display_df.iterrows():
                table_text += f"{date.strftime('%Y-%m-%d')}\t"
                table_text += f"{row['Close']:8.2f}\t"
                
                if pd.notna(row['CV']):
                    table_text += f"{row['CV']:5.2f}\t"
                else:
                    table_text += "  ---\t"
                    
                if pd.notna(row['Range_Ratio_MA']):
                    table_text += f"{row['Range_Ratio_MA']:6.2f}\t"
                else:
                    table_text += "   ---\t"
                    
                if pd.notna(row['ATR_Ratio']):
                    table_text += f"{row['ATR_Ratio']:5.2f}\t"
                else:
                    table_text += "  ---\t"
                    
                if pd.notna(row['Convergence_Score']):
                    table_text += f"{row['Convergence_Score']*100:5.1f}\t"
                else:
                    table_text += "  ---\t"
                
                if row['Is_Converged'] == 1:
                    table_text += " ○\n"
                else:
                    table_text += " ×\n"
            
            table_text += "=" * 85 + "\n"
            print(f"データテーブルテキスト作成完了: {len(table_text)}文字")
            window["-TABLE-"].update(table_text)
            print("データテーブルタブ更新完了")
            
            # 統計サマリー更新
            print("統計サマリーの表示処理を開始...")
            print(f"サマリーテキスト作成完了: {len(summary_text)}文字")
            window["-SUMMARY-"].update(summary_text)
            print("統計サマリータブ更新完了")
            
            # ウィジェットを明示的に再描画
            window["-PERIODS-"].Widget.update()
            window["-TABLE-"].Widget.update()
            window["-SUMMARY-"].Widget.update()
            
            # エクスポートボタン有効化
            window["-EXPORT-"].update(disabled=False)
            window["-EXPORT_PERIODS-"].update(disabled=False)
            window["-EXPORT_SUMMARY-"].update(disabled=False)
            
            # ステータス更新
            window["-STATUS-"].update(f"✓ {ticker} の分析が完了しました")
            
            # 画面を強制的に更新
            window.refresh()
            print("すべてのタブ更新処理完了")
        
        # テストデータボタン
        if event == "-TEST-":
            # テスト用の簡単なデータを作成
            test_dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
            test_data = {
                'Close': np.random.normal(1000, 50, 50),
                'High': np.random.normal(1020, 50, 50),
                'Low': np.random.normal(980, 50, 50),
                'Volume': np.random.randint(100000, 1000000, 50)
            }
            df = pd.DataFrame(test_data, index=test_dates)
            df = calculate_convergence_metrics(df, window=10)
            
            # テスト用の収束期間を作成
            convergence_periods = [
                {'start': '2024-01-10', 'end': '2024-01-15', 'duration': 6, 'avg_cv': 1.5},
                {'start': '2024-01-25', 'end': '2024-01-30', 'duration': 6, 'avg_cv': 1.8}
            ]
            
            # テスト統計サマリーを作成
            summary_text = "=" * 70 + "\n"
            summary_text += "     株価収束分析レポート - TEST\n"
            summary_text += "=" * 70 + "\n\n"
            
            summary_text += f"""【基本統計量】
  変動係数（CV）
    平均:     {df['CV'].mean():6.2f}%
    最小:     {df['CV'].min():6.2f}%
    最大:     {df['CV'].max():6.2f}%
    中央値:   {df['CV'].median():6.2f}%
    標準偏差: {df['CV'].std():6.2f}%

【収束統計】
  総収束日数:     {df['Is_Converged'].sum():4d}日
  収束率:         {df['Is_Converged'].sum() / len(df) * 100:6.1f}%
  検出期間数:     {len(convergence_periods):4d}期間
  最長収束期間:   {max([p['duration'] for p in convergence_periods], default=0):4d}日
  平均収束期間:   {np.mean([p['duration'] for p in convergence_periods]) if convergence_periods else 0:6.1f}日
  平均収束スコア: {df['Convergence_Score'].mean() * 100:6.1f}

【分析情報】
  銘柄コード: TEST
  分析期間:   50日
  データ数:   {len(df)}日分
  開始日:     {df.index[0].strftime('%Y-%m-%d')}
  終了日:     {df.index[-1].strftime('%Y-%m-%d')}

【パラメータ設定】
  計算ウィンドウ: 10日
  CV閾値:        2.0%
  レンジ閾値:    2.0%
  最小連続日数:  5日
"""
            
            summary_text += "\n" + "=" * 70 + "\n"
            summary_text += f"分析実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            
            # データ保存
            current_df = df
            current_ticker = "TEST"
            current_periods = convergence_periods
            current_summary = summary_text
            
            # メトリクス更新
            window["-METRIC1-"].update(f"{len(df)}日")
            window["-METRIC2-"].update(f"{df['Is_Converged'].sum()}日")
            window["-METRIC3-"].update(f"{df['Is_Converged'].sum() / len(df) * 100:.1f}%")
            window["-METRIC4-"].update(f"{len(convergence_periods)}件")
            
            # グラフ更新
            fig = create_convergence_plot(df, "TEST", convergence_periods)
            
            # 図を一時的にバイト列として保存
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=85, bbox_inches='tight', 
                       facecolor='white', edgecolor='none', pad_inches=0.1)
            buf.seek(0)
            
            # Imageウィジェットに表示
            window["-CANVAS-"].update(data=buf.read())
            plt.close(fig)  # メモリリーク防止
            
            # 収束期間表示
            print("テストデータ: 収束期間の表示処理を開始...")
            periods_text = "=" * 70 + "\n"
            periods_text += "期間\t開始日\t\t終了日\t\t日数\t平均CV(%)\n"
            periods_text += "=" * 70 + "\n"
            
            total_days = 0
            for i, period in enumerate(convergence_periods, 1):
                periods_text += f" {i:2d}\t{period['start']}\t{period['end']}\t"
                periods_text += f"{period['duration']:3d}日\t{period['avg_cv']:6.2f}%\n"
                total_days += period['duration']
            
            periods_text += "-" * 70 + "\n"
            periods_text += f"合計: {len(convergence_periods)}期間、{total_days}日間の収束\n"
            
            print(f"テストデータ: 収束期間テキスト作成完了: {len(periods_text)}文字")
            window["-PERIODS-"].update(periods_text)
            window["-PERIOD_COUNT-"].update(
                f"✓ 合計 {len(convergence_periods)} 期間が検出されました（総収束日数: {total_days}日）"
            )
            print("テストデータ: 収束期間タブ更新完了")
            
            # データテーブル更新
            print("テストデータ: データテーブルの表示処理を開始...")
            table_text = "=" * 85 + "\n"
            table_text += "日付\t\t終値\tCV(%)\tレンジ(%)\tATR(%)\tスコア\t収束\n"
            table_text += "=" * 85 + "\n"
            
            display_df = df[['Close', 'CV', 'Range_Ratio_MA', 'ATR_Ratio', 
                            'Convergence_Score', 'Is_Converged']].tail(20)
            
            for date, row in display_df.iterrows():
                table_text += f"{date.strftime('%Y-%m-%d')}\t"
                table_text += f"{row['Close']:8.2f}\t"
                
                if pd.notna(row['CV']):
                    table_text += f"{row['CV']:5.2f}\t"
                else:
                    table_text += "  ---\t"
                    
                if pd.notna(row['Range_Ratio_MA']):
                    table_text += f"{row['Range_Ratio_MA']:6.2f}\t"
                else:
                    table_text += "   ---\t"
                    
                if pd.notna(row['ATR_Ratio']):
                    table_text += f"{row['ATR_Ratio']:5.2f}\t"
                else:
                    table_text += "  ---\t"
                    
                if pd.notna(row['Convergence_Score']):
                    table_text += f"{row['Convergence_Score']*100:5.1f}\t"
                else:
                    table_text += "  ---\t"
                
                if row['Is_Converged'] == 1:
                    table_text += " ○\n"
                else:
                    table_text += " ×\n"
            
            table_text += "=" * 85 + "\n"
            print(f"テストデータ: データテーブルテキスト作成完了: {len(table_text)}文字")
            window["-TABLE-"].update(table_text)
            print("テストデータ: データテーブルタブ更新完了")
            
            # 統計サマリー更新
            print("テストデータ: 統計サマリーの表示処理を開始...")
            print(f"テストデータ: サマリーテキスト作成完了: {len(summary_text)}文字")
            window["-SUMMARY-"].update(summary_text)
            print("テストデータ: 統計サマリータブ更新完了")
            
            # ウィジェットを明示的に再描画
            window["-PERIODS-"].Widget.update()
            window["-TABLE-"].Widget.update()
            window["-SUMMARY-"].Widget.update()
            
            # エクスポートボタン有効化
            window["-EXPORT-"].update(disabled=False)
            window["-EXPORT_PERIODS-"].update(disabled=False)
            window["-EXPORT_SUMMARY-"].update(disabled=False)
            
            # ステータス更新
            window["-STATUS-"].update("✓ テストデータの分析が完了しました")
            
            # 画面を強制的に更新
            window.refresh()
            print("テストデータ: すべてのタブ更新処理完了")
        
        # CSVエクスポート（全データ）
        if event == "-EXPORT-" and current_df is not None:
            filename = eg.popup_get_file(
                "保存先を選択",
                save_as=True,
                default_extension=".csv",
                file_types=(("CSV Files", "*.csv"), ("All Files", "*.*")),
                default_path=f"{current_ticker}_convergence_analysis.csv"
            )
            
            if filename:
                try:
                    current_df.to_csv(filename, encoding='utf-8-sig')
                    eg.popup(f"ファイルを保存しました:\n{filename}", title="成功")
                except Exception as e:
                    eg.popup_error(f"保存エラー:\n{str(e)}")
        
        # 期間詳細をCSVエクスポート
        if event == "-EXPORT_PERIODS-" and current_periods is not None:
            filename = eg.popup_get_file(
                "収束期間データを保存",
                save_as=True,
                default_extension=".csv",
                file_types=(("CSV Files", "*.csv"), ("All Files", "*.*")),
                default_path=f"{current_ticker}_convergence_periods.csv"
            )
            
            if filename:
                try:
                    periods_df = pd.DataFrame(current_periods)
                    periods_df.to_csv(filename, index=False, encoding='utf-8-sig')
                    eg.popup(f"収束期間データを保存しました:\n{filename}", title="成功")
                except Exception as e:
                    eg.popup_error(f"保存エラー:\n{str(e)}")
        
        # サマリーをテキストエクスポート
        if event == "-EXPORT_SUMMARY-" and current_summary is not None:
            filename = eg.popup_get_file(
                "統計サマリーを保存",
                save_as=True,
                default_extension=".txt",
                file_types=(("Text Files", "*.txt"), ("All Files", "*.*")),
                default_path=f"{current_ticker}_summary.txt"
            )
            
            if filename:
                try:
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(current_summary)
                    eg.popup(f"統計サマリーを保存しました:\n{filename}", title="成功")
                except Exception as e:
                    eg.popup_error(f"保存エラー:\n{str(e)}")
    
    window.close()

if __name__ == "__main__":
    main()