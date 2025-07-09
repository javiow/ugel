import yfinance as yf
import pandas as pd
from tqdm import tqdm

def load_ticker_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    특정 종목(symbol)의 주가 데이터를 불러오는 함수
    :param symbol: 티커(ex. 'AAPL', '005930.KS')
    :param start: 시작일자 (ex. '2024-01-01')
    :param end: 종료일자 (ex. '2025-01-01')
    :return: OHLCV 데이터프레임
    """
    df = yf.download(symbol, start=start, end=end)
    df = df.reset_index()  # 인덱스를 날짜 컬럼으로
    return df


def load_tickers_data(symbols: list, start: str, end: str):
    """
    여러 종목의 주가 데이터를 딕셔너리 형태로 불러오는 함수
    :param symbols: 종목 티커 리스트 (ex. ['AAPL', 'MSFT', 'GOOG'])
    :param start: 시작일자
    :param end: 종료일자
    :return: {티커: 데이터프레임} 형태의 딕셔너리, 실패한 종목 리스트
    """

    data = {}
    fail_symbols = []

    for symbol in tqdm(symbols):
        try:
            df = yf.download(symbol, start=start, end=end)
            if not df.empty:
                data[symbol] = df.reset_index()
        except Exception as e:
            print(f"[{symbol}] 데이터 수집 실패: {e}")
            fail_symbols.append(symbol)

    return data, fail_symbols

def get_sp500_symbols():
    """
    S&P 500 지수에 포함된 종목의 티커를 가져오는 함수
    :return: S&P 500 종목 티커 리스트
    """
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    df = pd.read_html(url)[0]
    symbols = df['Symbol'].str.replace('.', '-', regex=False).tolist()
    return symbols

def get_nasdaq100_symbols():
    """
    NASDAQ-100 지수에 포함된 종목의 티커를 가져오는 함수
    :return: NASDAQ-100 종목 티커 리스트
    """
    url = 'https://en.wikipedia.org/wiki/NASDAQ-100'
    df = pd.read_html(url)[4]  # 실제 테이블 번호는 print로 먼저 확인
    symbols = df['Ticker'].str.replace('.', '-', regex=False).tolist()
    return symbols

def check_data_quality(df: pd.DataFrame, symbol: str):
    """
    종목 데이터프레임의 품질 점검 결과 출력
    :param df: OHLCV 데이터프레임 (Date 컬럼 포함)
    :param symbol: 종목 티커명
    """
    print(f"\n=== [{symbol}] 데이터 품질 점검 ===")
    # 1. 상장일(최초날짜), 상장폐지일(마지막날짜)
    if 'Date' in df.columns:
        start_date = df['Date'].min()
        end_date = df['Date'].max()
    else:
        df = df.reset_index()
        start_date = df['Date'].min()
        end_date = df['Date'].max()
    print(f"상장일(데이터 시작): {start_date}")
    print(f"상장폐지일(데이터 끝): {end_date}")

    # 2. 결측치 개수 (각 컬럼별)
    na_counts = df.isna().sum()
    na_exist = na_counts.any()
    print("\n결측치 존재 여부:", "있음" if na_exist else "없음")
    print("컬럼별 결측치 개수:")
    print(na_counts)

    # 3. 전체 결측치 행 개수
    na_rows = df.isna().any(axis=1).sum()
    print(f"\n결측치 포함 행 개수: {na_rows} / 전체 {len(df)}개 행")