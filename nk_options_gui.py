# nk_options_gui.py
# TkEasyGUI + Selenium：QRI（日経225オプション）スクレイパ GUI
# - 限月コンボは起動時に現在月から4か月分(YYYY-MM)を常に表示
# - タブが取れたらクリック/URLで確実遷移、失敗時は /jpx/nkopm/ の規則にフォールバック
# - 上テーブルは“格子線風”（数値右寄せ＋交互色）で視認性UP
# - 上下の間にスライダーで高さ比率（上%）を調整可能
# - ITM/OTM の価格レンジ指定で割安候補を提案

import os
import math, re, time, subprocess
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import yfinance as yf
import TkEasyGUI as eg  # pip install TkEasyGUI
import tkinter as tk
from tkinter import ttk

from selenium import webdriver
#from selenium.webdriver.common.By import By
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from urllib.parse import urlparse, urljoin

# ---- ログ抑制（任意）----
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # TFLiteのINFO/WARN非表示

# ================= 定数 =================
JPX_DERIV_QUOTES = "https://www.jpx.co.jp/english/markets/derivatives/quotes/index.html"
UNDERLYING_DEFAULT = "^N225"
DEFAULT_RF = 0.01
DEFAULT_Q  = 0.0

# ================= 数学/BS =================
SQRT_2PI = math.sqrt(2.0 * math.pi)
def _norm_cdf(x): return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
def _norm_pdf(x): return math.exp(-0.5 * x * x) / SQRT_2PI
def _d1_d2(S,K,r,q,sigma,T):
    if sigma<=0 or T<=0 or S<=0 or K<=0: return np.nan, np.nan
    v=sigma*math.sqrt(T); d1=(math.log(S/K)+(r-q+0.5*sigma*sigma)*T)/v; d2=d1-v; return d1,d2
def bs_price(cp,S,K,r,q,sigma,T):
    d1,d2=_d1_d2(S,K,r,q,sigma,T)
    if np.isnan(d1): return np.nan
    return S*math.exp(-q*T)*_norm_cdf(d1)-K*math.exp(-r*T)*_norm_cdf(d2) if cp.upper()=='C' else K*math.exp(-r*T)*_norm_cdf(-d2)-S*math.exp(-q*T)*_norm_cdf(-d1)
def greeks(cp,S,K,r,q,sigma,T):
    d1,d2=_d1_d2(S,K,r,q,sigma,T)
    if np.isnan(d1): return dict(delta=np.nan,gamma=np.nan,theta=np.nan,vega=np.nan,rho=np.nan)
    nd1=_norm_pdf(d1)
    if cp.upper()=='C':
        delta=math.exp(-q*T)*_norm_cdf(d1)
        theta=(-S*nd1*sigma*math.exp(-q*T)/(2*math.sqrt(T))
               - r*K*math.exp(-r*T)*_norm_cdf(d2)
               + q*S*math.exp(-q*T)*_norm_cdf(d1))
        rho=K*T*math.exp(-r*T)*_norm_cdf(d2)
    else:
        delta=-math.exp(-q*T)*_norm_cdf(-d1)
        theta=(-S*nd1*sigma*math.exp(-q*T)/(2*math.sqrt(T))
               + r*K*math.exp(-r*T)*_norm_cdf(-d2)
               - q*S*math.exp(-q*T)*_norm_cdf(-d1))
        rho=-K*T*math.exp(-r*T)*_norm_cdf(-d2)
    gamma=(math.exp(-q*T)*nd1)/(S*sigma*math.sqrt(T))
    vega=S*math.exp(-q*T)*nd1*math.sqrt(T)/100.0
    return dict(delta=delta,gamma=gamma,theta=theta/365.0,vega=vega,rho=rho/100.0)

# ================= 日付ユーティリティ =================
def third_friday(year:int, month:int)->date:
    d=date(year,month,1)
    while d.weekday()!=4: d+=timedelta(days=1)
    d+=timedelta(weeks=2); return d

def upcoming_months(n=12)->List[Tuple[str,date]]:
    today=date.today(); y,m=today.year,today.month; out=[]
    for i in range(n):
        mm=(m+i-1)%12+1; yy=y+(m+i-1)//12
        out.append((f"{yy}-{mm:02d}", third_friday(yy,mm)))
    return out

def month_from_any(label:str)->Optional[int]:
    if not label: return None
    m=re.search(r"(\d{4})-(\d{1,2})", label)
    if m: return int(m.group(2))
    m=re.search(r"(\d+)\s*月", label)
    if m: return int(m.group(1))
    return None

# ================= yfinance =================
def _flatten_yf(df:pd.DataFrame)->pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns=[c[0] if isinstance(c,tuple) else str(c) for c in df.columns]
    else:
        df.columns=[str(c) for c in df.columns]
    return df
def annualize_hist_vol(df_price:pd.DataFrame, window=30)->float:
    df_price=_flatten_yf(df_price)
    if "Close" not in df_price.columns: return np.nan
    ret=np.log(df_price["Close"]).diff().dropna()
    if len(ret)<5: return np.nan
    return float(ret.rolling(window).std().dropna().iloc[-1]*math.sqrt(252))

# ================= Selenium =================
def make_driver(headless=True):
    opts=Options()
    if headless: opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu"); opts.add_argument("--no-sandbox")
    opts.add_argument("--window-size=1600,1200")
    opts.add_argument("--lang=ja-JP,ja;q=0.9,en-US;q=0.8")
    opts.add_argument("--log-level=3"); opts.add_argument("--disable-logging")
    opts.add_experimental_option("excludeSwitches", ["enable-automation","enable-logging"])
    service = Service(ChromeDriverManager().install(), log_output=subprocess.DEVNULL)
    driver=webdriver.Chrome(service=service, options=opts)
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument",
        {"source":"Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"})
    return driver

def navigate_to_qri(driver):
    wait=WebDriverWait(driver,25)
    driver.get(JPX_DERIV_QUOTES)
    link=wait.until(EC.element_to_be_clickable(
        (By.XPATH,"//a[contains(.,'Options Quotes') or contains(@href,'svc.qri.jp')]")))
    link.click(); time.sleep(1)
    if len(driver.window_handles)>1: driver.switch_to.window(driver.window_handles[-1])
    for txt in ["I agree","I accept","同意","OK"]:
        els=driver.find_elements(By.XPATH,f"//*[contains(.,'{txt}')]")
        if els:
            try: els[0].click(); break
            except: pass
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR,"div.fixed > table.price-table")))
    time.sleep(0.4)

def get_month_tabs_with_href(driver)->Tuple[List[str], str, List[Dict[str,str]]]:
    tabs=[]
    items = driver.find_elements(By.XPATH, "//ul/li[a[contains(normalize-space(.),'月限月')]]")
    for li in items:
        a = li.find_element(By.TAG_NAME, "a")
        label = a.text.strip()
        href = a.get_attribute("href") or a.get_attribute("data-href") or a.get_attribute("onclick") or ""
        is_active = "active" in (li.get_attribute("class") or "")
        if label:
            tabs.append({"label": label, "href": href, "active": is_active})
    labels = [t["label"] for t in tabs]
    active_label = next((t["label"] for t in tabs if t["active"]), "")
    return labels, active_label, tabs

def _origin(url:str)->str:
    p=urlparse(url); return f"{p.scheme}://{p.netloc}"

def _base_nkopm(url:str)->Optional[str]:
    p=urlparse(url)
    m=re.match(r"^(.*?/jpx/nkopm)(?:/\d+)?/?$", p.path)
    if not m: return None
    return _origin(url)+m.group(1)+"/"

def ensure_month_page(driver, desired_label:str, tabs:List[Dict[str,str]]):
    if not tabs or not desired_label: return
    rec=next((t for t in tabs if t["label"]==desired_label), None)
    if not rec: return
    href=rec.get("href") or ""
    old_html=driver.page_source
    cur_url=driver.current_url
    if href and href.startswith("http"):
        driver.get(href)
    elif href and href!="javascript:void(0)":
        driver.get(urljoin(_origin(cur_url), href))
    else:
        els=driver.find_elements(By.XPATH, f"//a[contains(normalize-space(.),'{desired_label}')]")
        if els: driver.execute_script("arguments[0].click();", els[0])
    wait=WebDriverWait(driver,20)
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR,"div.fixed > table.price-table")))
    for _ in range(40):
        act=driver.find_elements(By.XPATH,"//ul/li[contains(@class,'active')]/a[contains(.,'月限月')]")
        if act and act[0].text.strip()==desired_label: break
        time.sleep(0.15)
    for _ in range(40):
        if driver.page_source!=old_html: break
        time.sleep(0.1)
    time.sleep(0.2)

def ensure_month_page_fallback(driver, desired_label_or_yyyymm:str):
    base=_base_nkopm(driver.current_url)
    if not base: return
    mm=month_from_any(desired_label_or_yyyymm)
    if not mm: return
    diff=(mm - date.today().month) % 12
    suffix="" if diff==0 else str(diff)
    driver.get(base + suffix)
    WebDriverWait(driver,20).until(EC.presence_of_element_located((By.CSS_SELECTOR,"div.fixed > table.price-table")))
    time.sleep(0.2)

# ================= 単一テーブル抽出（17列） =================
def _num_first(s:str):
    if not s: return None
    m=re.search(r"-?\d[\d,]*\.?\d*", s)
    try: return float(m.group(0).replace(",","")) if m else None
    except: return None
def _pct_first(s:str):
    if not s: return None
    m=re.search(r"-?\d+(?:\.\d+)?", s)
    try: return float(m.group(0))/100.0 if m else None
    except: return None
def _pair_bid_ask_from_two_lines(cell_text:str):
    lines=[t.strip() for t in cell_text.splitlines() if t.strip()!=""]
    ask=_num_first(lines[0]) if len(lines)>=1 else None
    bid=_num_first(lines[1]) if len(lines)>=2 else None
    return bid, ask
def _scroll_tbody(driver, tbody):
    driver.execute_script("""
        const tb = arguments[0];
        if (!tb) return;
        tb.scrollTop = 0;
        let H = tb.scrollHeight;
        for (let y=0; y<=H; y+=600) { tb.scrollTop = y; }
        tb.scrollTop = H;
    """, tbody)
    time.sleep(0.2)

def scrape_qri_single_table_current(driver):
    wait=WebDriverWait(driver,25)
    table=wait.until(EC.presence_of_element_located((By.CSS_SELECTOR,"div.fixed > table.price-table")))
    tbody=table.find_element(By.CSS_SELECTOR,"tbody.price-info-scroll")
    for _ in range(3): _scroll_tbody(driver, tbody)
    rows=tbody.find_elements(By.CSS_SELECTOR,"tr.row-num")
    call_rows, put_rows=[], []
    for tr in rows:
        tds=tr.find_elements(By.CSS_SELECTOR,"td")
        if len(tds)<17: continue
        strike=_num_first(tds[8].text)
        # CALL（左8）
        c_bid,c_ask=_pair_bid_ask_from_two_lines(tds[4].text)
        c_last=_num_first(tds[7].text)
        c_iv=_pct_first(tds[5].text)
        c_vol=_num_first(tds[2].text)
        c_oi=_num_first(tds[1].text)
        call_rows.append({"type":"C","strike":strike,"bid":c_bid,"ask":c_ask,"last":c_last,"iv":c_iv,"volume":c_vol,"open_interest":c_oi})
        # PUT（右8）
        p_bid,p_ask=_pair_bid_ask_from_two_lines(tds[12].text)
        p_last=_num_first(tds[9].text)
        p_iv=_pct_first(tds[11].text)
        p_vol=_num_first(tds[14].text)
        p_oi=_num_first(tds[15].text)
        put_rows.append({"type":"P","strike":strike,"bid":p_bid,"ask":p_ask,"last":p_last,"iv":p_iv,"volume":p_vol,"open_interest":p_oi})
    return pd.DataFrame(call_rows), pd.DataFrame(put_rows)

# ================= データ加工 =================
def parse_float(x):
    try:
        if isinstance(x,str): x=x.replace(",","")
        return float(x)
    except: return np.nan
def mid_price(row):
    b,a,last=row.get("bid",np.nan),row.get("ask",np.nan),row.get("last",np.nan)
    b=parse_float(b); a=parse_float(a); last=parse_float(last)
    if not np.isnan(b) and not np.isnan(a) and a>=b: return (a+b)/2.0
    if not np.isnan(last): return last
    return np.nan

def compute_theo_and_greeks(df:pd.DataFrame, cp:str, S:float, r:float, q:float, T:float, fallback_sigma:float):
    out=df.copy()
    for c in ["bid","ask","last","strike","iv","volume","open_interest"]:
        if c in out.columns: out[c]=out[c].apply(parse_float)
    if "strike" not in out.columns or out["strike"].notna().sum()<max(5,int(len(out)*0.2)): return out
    def pick_sigma(x):
        if pd.isna(x): return fallback_sigma
        v=float(x); return v/100.0 if v>3 else v
    sigmas=out["iv"].apply(pick_sigma) if "iv" in out.columns else pd.Series([fallback_sigma]*len(out))
    mids=out.apply(mid_price, axis=1)
    theos=[]; deltas=[]; gammas=[]; thetas=[]; vegas=[]; rhos=[]
    for k,sig in zip(out["strike"], sigmas):
        if pd.isna(k) or pd.isna(sig) or sig<=0 or pd.isna(S) or T<=0:
            theos.append(np.nan); deltas.append(np.nan); gammas.append(np.nan); thetas.append(np.nan); vegas.append(np.nan); rhos.append(np.nan); continue
        theo=bs_price(cp,S,float(k),r,q,float(sig),T)
        g=greeks(cp,S,float(k),r,q,float(sig),T)
        theos.append(theo); deltas.append(g["delta"]); gammas.append(g["gamma"]); thetas.append(g["theta"]); vegas.append(g["vega"]); rhos.append(g["rho"])
    out["mid"]=mids; out["theo"]=pd.Series(theos)
    out["dev_pct"]=np.where(out["mid"]>0,(out["theo"]-out["mid"])/out["mid"]*100.0,np.nan)
    out["delta"]=deltas; out["gamma"]=gammas; out["theta"]=thetas; out["vega"]=vegas; out["rho"]=rhos
    for c in ["mid","theo","dev_pct","iv","delta","gamma","theta","vega","rho","bid","ask","last","strike"]:
        if c in out.columns: out[c]=out[c].astype(float).round(4)
    return out

def suggest_undervalued(df:pd.DataFrame, topn=5, min_vol=10)->pd.DataFrame:
    if df is None or df.empty or "dev_pct" not in df.columns: return pd.DataFrame()
    d=df.copy()
    if "volume" in d.columns: d=d[d["volume"].fillna(0)>=min_vol]
    return d.sort_values("dev_pct", ascending=False).head(topn)

# ================= 日本語見出し =================
JP_HEADERS = {
    "strike": "権利行使価格",
    "bid": "買気配",
    "ask": "売気配",
    "last": "現在値",
    "mid": "ミッド",
    "theo": "理論価格",
    "dev_pct": "乖離(%)",
    "iv": "IV",
    "volume": "取引高",
    "open_interest": "建玉残",
    "delta": "デルタ",
    "gamma": "ガンマ",
    "theta": "セータ/日",
    "vega": "ベガ",
    "rho": "ロー",
}

# ================= ATM＆レンジユーティリティ =================
def nearest_atm_strike(call_df: pd.DataFrame, put_df: pd.DataFrame, S: float | None) -> float | None:
    """原資産 S に最も近い strike を ATM とみなす。
    S が取れない場合は全ストライクの中央値近傍を採用。
    """
    series_list = []
    for df in (call_df, put_df):
        if isinstance(df, pd.DataFrame) and "strike" in df.columns:
            series_list.append(pd.to_numeric(df["strike"], errors="coerce"))

    if not series_list:
        return None

    # ★ インデックス振り直し＆NaN除去（重複ラベル対策）
    all_strikes = pd.concat(series_list, ignore_index=True).dropna()
    if all_strikes.empty:
        return None

    # 目標値：S が取れれば S、なければ中央値
    if S is not None and isinstance(S, (int, float, np.floating)) and not np.isnan(S):
        target = float(S)
    else:
        target = float(all_strikes.median())

    # ★ 位置で最小差の要素を選ぶ（iloc）
    diffs = (all_strikes - target).abs().to_numpy()
    idx_pos = int(np.nanargmin(diffs))
    return float(all_strikes.iloc[idx_pos])


def filter_by_range(df: pd.DataFrame, cp: str, atm: float, rng_itm: float, rng_otm: float) -> pd.DataFrame:
    """
    例: ATM=40000, ITM=2000, OTM=5000
      - Call: [38000, 45000]
      - Put : [35000, 42000]
    """
    if df is None or df.empty or "strike" not in df.columns:
        return pd.DataFrame()

    strikes = pd.to_numeric(df["strike"], errors="coerce")

    if cp.upper() == "C":
        lo, hi = atm - rng_itm, atm + rng_otm
    else:
        lo, hi = atm - rng_otm, atm + rng_itm

    mask = (strikes >= lo) & (strikes <= hi)
    out = df.loc[mask].copy()
    return out


# ================= GUI =================
def build_layout():
    # 起動時に必ず4か月(YYYY-MM)を表示
    labels=[m for m,_ in upcoming_months(4)]
    layout=[
        [eg.Text("Nikkei225 Options Screener (QRIスクレイプ＋BSモデル)")],
        [eg.Text("限月（サイトのタブから）"),
         eg.Combo(values=labels, default_value=labels[0], key="-MONTH-", readonly=True, size=(12,1)),
         eg.Text("償還日(YYYY-MM-DD可)"), eg.Input("", key="-EXPDATE-", size=(12,1)),
         eg.Text("原資産"), eg.Input(UNDERLYING_DEFAULT, key="-SYM-", size=(10,1)),
         eg.Text("無リスク金利"), eg.Input(str(DEFAULT_RF), key="-RF-", size=(6,1)),
         eg.Text("配当q"), eg.Input(str(DEFAULT_Q), key="-Q-", size=(6,1)),
         eg.Checkbox("ブラウザ表示", key="-SHOW-", default=False)],
        [eg.Button("データ取得", key="-FETCH-"),
         eg.Button("乖離で降順ソート", key="-SORT-"),
         eg.Text("出来高≧"), eg.Input("10", key="-MINVOL-", size=(6,1)),
         eg.Text("ITM側"), eg.Input("2000", key="-ITM-", size=(6,1)),
         eg.Text("OTM側"), eg.Input("5000", key="-OTM-", size=(6,1)),
         eg.Button("割安候補を提案", key="-SUGGEST-"),
         eg.Button("CSVエクスポート", key="-EXPORT-"),
         eg.Button("終了", key="-EXIT-")],
        # 薄型スライダー（高さは size の第2引数）
        [eg.Text("上下比率（上%）"),
         eg.Slider(range=(30, 90), default_value=50, resolution=5,
                   orientation="h", key="-SPLIT-", enable_events=True,
                   size=(24, 2))],
        [eg.HSeparator()],
        # 上：テーブル（縦に伸びる）
        [eg.TabGroup([[
            eg.Tab("Call", [[eg.Table(values=[[]], headings=[], key="-CALLTAB-", height=18, expand_x=True, expand_y=True)]]),
            eg.Tab("Put",  [[eg.Table(values=[[]], headings=[], key="-PUTTAB-",  height=18, expand_x=True, expand_y=True)]])
        ]], key="-TABS-", expand_x=True, expand_y=True)],
        [eg.HSeparator()],
        # 下：ログ（横いっぱい）
        [eg.Frame("割安候補（Top5）", [[eg.Multiline(size=(160,10), key="-IDEAS-", autoscroll=True, expand_x=True)]], expand_x=True)]
    ]
    return layout

@dataclass
class AppState:
    call_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    put_df:  pd.DataFrame = field(default_factory=pd.DataFrame)
    last_export_paths: List[str] = field(default_factory=list)
    underlying_S: float = float("nan")

def apply_table_style(win, key):
    """Treeviewの見やすさ強化：右寄せ・モノスペ・交互色（格子線風）"""
    tree = win[key].widget  # ttk.Treeview
    if not isinstance(tree, ttk.Treeview): return
    style = ttk.Style(tree)
    style.configure("Grid.Treeview", font=("Consolas", 10), rowheight=20)
    tree.configure(style="Grid.Treeview")
    # 全列を右寄せ
    for c in tree["columns"]:
        tree.column(c, anchor="e")
    # シマシマ
    tree.tag_configure("odd",  background="#f7f7f7")
    tree.tag_configure("even", background="#ffffff")
    for i, item in enumerate(tree.get_children("")):
        tree.item(item, tags=("odd" if i % 2 else "even",))

def update_table(win, key, df:pd.DataFrame, show_cols:Optional[List[str]]=None):
    if df is None or df.empty:
        win[key].update(values=[[]], headings=[]); return
    if show_cols is None:
        order=["strike","bid","ask","last","mid","theo","dev_pct","iv","volume","open_interest","delta","gamma","theta","vega","rho"]
        show_cols=[c for c in order if c in df.columns]
    vals=df[show_cols].replace({np.nan:""}).values.tolist()
    headings_jp=[JP_HEADERS.get(c,c) for c in show_cols]
    win[key].update(values=vals, headings=headings_jp)
    apply_table_style(win, key)

def fetch_underlying_price(sym:str)->float:
    try:
        data=yf.download(sym, period="10d", interval="1d", progress=False, auto_adjust=False)
        data=_flatten_yf(data)
        if data.empty or "Close" not in data.columns: return np.nan
        return float(data["Close"].dropna().iloc[-1])
    except: return np.nan
def fetch_histvol(sym:str)->float:
    try:
        data=yf.download(sym, period="6mo", interval="1d", progress=False, auto_adjust=False)
        if data.empty: return np.nan
        return annualize_hist_vol(data, window=30)
    except: return np.nan

def run_fetch(headless, selected_month_ym, exp_override, sym, r_str, q_str, window, state:AppState):
    window["-IDEAS-"].print("スクレイプ開始…")
    drv=None
    try:
        mm = month_from_any(selected_month_ym) or date.today().month
        drv=make_driver(headless=headless)
        navigate_to_qri(drv)

        labels, active, tabs = get_month_tabs_with_href(drv)
        desired_label_jp = f"{mm}月限月"
        if tabs:
            ensure_month_page(drv, desired_label_jp, tabs)
        else:
            ensure_month_page_fallback(drv, selected_month_ym)

        # 満期日
        try:
            exp_date=datetime.strptime(exp_override,"%Y-%m-%d").date() if exp_override else None
        except:
            exp_date=None
        if exp_date is None:
            today=date.today()
            yy = today.year if mm>=today.month else today.year+1
            exp_date=third_friday(yy, mm)

        call_raw, put_raw = scrape_qri_single_table_current(drv)

        S=fetch_underlying_price(sym); hv=fetch_histvol(sym)
        try: r=float(r_str)
        except: r=DEFAULT_RF
        try: q=float(q_str)
        except: q=DEFAULT_Q

        T_days=max((exp_date - date.today()).days, 1); T=T_days/365.0
        fallback_sigma = hv if not np.isnan(hv) and hv>0 else 0.2

        call_df=compute_theo_and_greeks(call_raw,'C',S,r,q,T,fallback_sigma)
        put_df =compute_theo_and_greeks(put_raw, 'P',S,r,q,T,fallback_sigma)

        state.call_df, state.put_df = call_df, put_df
        state.underlying_S = float(S) if isinstance(S,(int,float,np.floating)) and not np.isnan(S) else float("nan")
        update_table(window,"-CALLTAB-",call_df); update_table(window,"-PUTTAB-",put_df)

        S_disp=f"{float(S):.2f}" if isinstance(S,(int,float,np.floating)) and not np.isnan(S) else str(S)
        window["-IDEAS-"].print(f"取得完了: S={S_disp}, T={T_days}日, r={r}, q={q}, 予備σ={fallback_sigma:.3f}")
        window["-IDEAS-"].print(f"行数: Call={len(call_df)}, Put={len(put_df)}")

        def chk(df,name):
            need=["strike","bid","ask","last","iv"]; miss=[c for c in need if c not in df.columns]
            if miss: window["-IDEAS-"].print(f"[WARN] {name} 欠落列: {miss}")
            if "strike" in df.columns:
                filled=int(df["strike"].notna().sum())
                window["-IDEAS-"].print(f"[INFO] {name} strike 非NaN {filled}/{len(df)}")
        chk(call_df,"CALL"); chk(put_df,"PUT")
        window["-IDEAS-"].print("「割安候補を提案」を押すとTop5を提案します。")

    except Exception as e:
        window["-IDEAS-"].print(f"[ERROR] {e}")
    finally:
        if drv:
            try: drv.quit()
            except: pass

# ---- 上下比率（上%）に応じて高さを調整 ----
def adjust_split(win, percent=70):
    percent = max(30, min(90, int(percent)))
    # 30% → 10行, 90% → 28行 に線形マッピング（調整可）
    top_rows = int(10 + (percent - 30) * (28 - 10) / (90 - 30))
    top_rows = max(8, min(30, top_rows))
    try: win["-CALLTAB-"].widget.configure(height=top_rows)
    except: pass
    try: win["-PUTTAB-"].widget.configure(height=top_rows)
    except: pass
    bottom_h = max(6, 36 - top_rows)
    try: win["-IDEAS-"].widget.configure(height=bottom_h)
    except: pass



def main():
    for t in ("vista","xpnative","clam","default","alt","classic"):
        try: eg.theme(t); break
        except: continue
    state=AppState()
    layout=build_layout()
    window=eg.Window("NK Options Screener", layout, resizable=True, finalize=True, size=(1600, 1000))

    # 起動時に初期比率を適用（70%）
    adjust_split(window, 70)

    while True:
        event, values = window.read()
        if event in (eg.WINDOW_CLOSED, "-EXIT-"): break

        if event=="-SPLIT-":
            adjust_split(window, values["-SPLIT-"])

        if event=="-FETCH-":
            run_fetch(not values["-SHOW-"], values["-MONTH-"], values["-EXPDATE-"],
                      values["-SYM-"], values["-RF-"], values["-Q-"], window, state)

        if event=="-SORT-":
            for key, df in [("-CALLTAB-", state.call_df), ("-PUTTAB-", state.put_df)]:
                if df is not None and not df.empty and "dev_pct" in df.columns:
                    df.sort_values("dev_pct", ascending=False, inplace=True, kind="mergesort")
                    update_table(window, key, df)

        if event=="-SUGGEST-":
            try: min_vol=int(values["-MINVOL-"])
            except: min_vol=10
            try: rng_itm=float(values["-ITM-"])
            except: rng_itm=2000.0
            try: rng_otm=float(values["-OTM-"])
            except: rng_otm=5000.0

            atm = nearest_atm_strike(state.call_df, state.put_df, state.underlying_S)
            if atm is None:
                window["-IDEAS-"].update("ATM を推定できません（strike 列が不足または空）。\n")
                continue

            call_r = filter_by_range(state.call_df, "C", atm, rng_itm, rng_otm)
            put_r  = filter_by_range(state.put_df,  "P", atm, rng_itm, rng_otm)

            call_pick = suggest_undervalued(call_r, topn=5, min_vol=min_vol)
            put_pick  = suggest_undervalued(put_r,  topn=5, min_vol=min_vol)

            c_lo, c_hi = atm - rng_itm, atm + rng_otm
            p_lo, p_hi = atm - rng_otm, atm + rng_itm

            txt=[]
            txt.append(f"[INFO] ATM推定: {atm:.0f} / レンジ: CALL[{c_lo:.0f}～{c_hi:.0f}] PUT[{p_lo:.0f}～{p_hi:.0f}] / 出来高≧{min_vol}")
            if not call_pick.empty:
                txt.append("\n[CALL 割安候補 Top5]")
                txt.append(call_pick[["strike","mid","theo","dev_pct","iv","volume","delta","theta"]].to_string(index=False))
            if not put_pick.empty:
                txt.append("\n[PUT 割安候補 Top5]")
                txt.append(put_pick[["strike","mid","theo","dev_pct","iv","volume","delta","theta"]].to_string(index=False))
            if len(txt) == 1:
                txt.append("\n条件に合う候補が見つかりませんでした（データ未取得/レンジ外/出来高不足）。")
            window["-IDEAS-"].update("\n".join(txt))

        if event=="-EXPORT-":
            paths=[]; ts=datetime.now().strftime("%Y%m%d_%H%M%S")
            if state.call_df is not None and not state.call_df.empty:
                p=f"call_{ts}.csv"; state.call_df.to_csv(p,index=False,encoding="utf-8-sig"); paths.append(p)
            if state.put_df is not None and not state.put_df.empty:
                p=f"put_{ts}.csv";  state.put_df.to_csv(p,index=False,encoding="utf-8-sig"); paths.append(p)
            eg.popup("CSVを書き出しました:\n"+"\n".join(paths) if paths else "出力できるデータがありません。")
    window.close()

if __name__=="__main__":
    main()
