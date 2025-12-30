import pandas as pd
import requests
import yfinance as yf
import pickle
import os

from datetime import datetime
from pathlib import Path
from io import StringIO
from tqdm import tqdm

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL  = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

def load_symdir(url: str) -> pd.DataFrame:
    txt = requests.get(url, timeout=30).text
    # 마지막 트레일러 라인 제거 (File Creation Time:)
    lines = [ln for ln in txt.splitlines() if not ln.startswith("File Creation Time:")]
    df = pd.read_csv(StringIO("\n".join(lines)), sep="|", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    return df

def sanitize_tickers(tickers):
    # pandas Series / list 모두 처리
    s = pd.Series(list(tickers))

    # 문자열화 + 결측 제거
    s = s.dropna().astype(str)

    # 앞뒤 공백 제거
    s = s.str.strip()

    # 빈 문자열 제거
    s = s[s.ne("")]

    # 'nan' 같은 문자열로 들어간 것도 제거
    s = s[~s.str.lower().isin(["nan", "none", "null"])]

    # 혹시 리스트/튜플 문자열화로 이상해진 케이스 방지(공백 포함 등)
    # 미국 주식/ETF 티커는 보통 대문자, '.', '-', '^' 등 일부 기호가 섞일 수 있음
    s = s[s.str.match(r"^[A-Za-z0-9\.\-^=]+$")]

    # 대문자 통일 + 중복 제거
    s = s.str.upper().drop_duplicates()

    return s.tolist()

def etf_filter(df, start_date=None, end_date=None):
    # 기준일은 "데이터 최신일"을 쓰는 게 백테스트/스냅샷에 안전
    ref_date = pd.Timestamp(df.index.max()).normalize()

    # start_date: 기준일로부터 20년 전
    if start_date is None:
        start = ref_date - pd.DateOffset(years=20)
    else:
        start = pd.Timestamp(start_date)

    # end_date: start_date로부터 6개월 뒤
    if end_date is None:
        end = start + pd.DateOffset(months=6)
    else:
        end = pd.Timestamp(end_date)

    # 1) [start~end] 구간에 유효값이 한 번이라도 있는 ETF
    has_df = df.loc[start:end].notna().any(axis=0)

    # 2) 최신일(데이터셋 마지막 날짜)까지 유효값이 있는 ETF
    max_date = df.index.max()
    has_latest = df.loc[max_date].notna()

    # 최종 필터
    keep = has_df & has_latest

    # 기존 코드의 반환부는 타입이 안 맞아 오류/오해 소지가 커서,
    # "통과한 티커 리스트"를 반환하는 형태로 최소 수정
    return keep['Close'].values[0]

def make_ticker_list(tickers_clean, start_date, end_date, sample_len=None):

    ticker_list = {}

    if sample_len is not None:
        import random
        # random.seed(42)
        tickers = random.sample(tickers_clean, sample_len)
    else:
        tickers = tickers_clean


    for ticker in tqdm(tickers):
        try:

            df = yf.download(
                ticker,
                start="2000-01-01",
                auto_adjust=True,   # 배당/분할 반영 (백테스트에 유리)
                progress=False,
                group_by="column"
            )

            if etf_filter(df, start_date=start_date, end_date=end_date):
                ticker_list[ticker] = df
                print(f"{ticker} added to ticker_list.")

        except:
            print(f"Error downloading data for {ticker}")

    qqq = yf.download(
                "QQQ",
                start="2000-01-01",
                auto_adjust=True,   # 배당/분할 반영 (백테스트에 유리)
                progress=False,
                group_by="column"
            )

    ief = yf.download(
                "IEF",
                start="2000-01-01",
                auto_adjust=True,   # 배당/분할 반영 (백테스트에 유리)
                progress=False,
                group_by="column"
            )

    ticker_list['QQQ'] = qqq
    ticker_list['IEF'] = ief

    return ticker_list

def save_df_dict_pickle(ticker_df: dict, filepath: str):
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(ticker_df, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_df_dict_pickle(filepath: str) -> dict:
    path = Path(filepath)
    with path.open("rb") as f:
        return pickle.load(f)

def make_daa_data(start_date="2004-01-01", end_date="2004-12-31", sample_len=500):

    nas = load_symdir(NASDAQ_LISTED_URL)
    oth = load_symdir(OTHER_LISTED_URL)

    # 티커 컬럼 표준화: nas는 Symbol, other는 ACT Symbol
    nas["ticker"] = nas["Symbol"]
    oth["ticker"] = oth["ACT Symbol"]

    all_df = pd.concat([nas, oth], ignore_index=True, sort=False)

    # ETF만 필터 + 테스트이슈 제거(권장)
    etf_df = all_df[all_df["ETF"].fillna("N").eq("Y")].copy()
    if "Test Issue" in etf_df.columns:
        etf_df = etf_df[etf_df["Test Issue"].fillna("N").eq("N")]

    # ticker 정리
    etf_df["ticker"] = (
        etf_df["ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
    )
    etf_df = etf_df[etf_df["ticker"].ne("")].drop_duplicates(subset=["ticker"]).reset_index(drop=True)

    tickers = etf_df["ticker"].tolist()

    print("ETF 티커 수:", len(tickers))

    tickers_clean = sanitize_tickers(tickers)
    ticker_list = make_ticker_list(tickers_clean, start_date=start_date, end_date=end_date, sample_len=sample_len)
    ticker_df = pd.DataFrame()

    for ticker in ticker_list:
        print(f"{ticker}: {ticker_list[ticker].shape}")
        ticker_df[ticker] = ticker_list[ticker]['Close']

    ticker_df.index.name = "date_time"

    now_date = datetime.now().strftime("%Y%m%d")

    os.makedirs("./data/daa", exist_ok=True)
    os.makedirs(f"./data/daa/{now_date}", exist_ok=True)

    save_df_dict_pickle(ticker_df, f"./data/daa/{now_date}/daa_etf_prices.pkl")
    save_df_dict_pickle(ticker_df, f"./data/daa/daa_etf_prices.pkl")

    print(f"DAA 데이터셋이 ./data/daa/{now_date}/daa_etf_prices.pkl 에 저장되었습니다.")

if __name__ == "__main__":
    make_daa_data(sample_len=100)