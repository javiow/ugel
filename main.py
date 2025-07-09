from benchmark import data_load
from strategy import sample_strategy

def main():

    # NASDAQ-100 종목 데이터 불러오기
    nasdaq100_symbols = data_load.get_nasdaq100_symbols()
    nasdaq100_data, fail_list = data_load.load_tickers_data(nasdaq100_symbols, start='2023-01-01', end='2024-01-01')

    # 결과 출력
    print("NASDAQ-100 종목 데이터 불러오기 완료")
    print("예시 데이터")
    print(nasdaq100_data['AAPL'].head())  # AAPL 데이터 예시 출력

    if fail_list:
        print("데이터 수집 실패 종목:")
        for symbol in fail_list:
            print(symbol)

    df = nasdaq100_data['AAPL']

    print("데이터 품질 평가: AAPL")
    print(data_load.check_data_quality(df, 'Apple'))

    # 이동평균선 골든크로스/데드크로스 전략 적용
    strategy_params = {'short': 20, 'long': 60}

    df_st = sample_strategy.sample_strategy(df, params=strategy_params)
    print("이동평균선 골든크로스/데드크로스 전략 적용 완료")
    print(df_st.head())

    # 매수/매도 인덱스 추출
    index_info = sample_strategy.get_short_long_index(df_st)
    print("매수/매도 인덱스 추출 완료")
    print("매수 인덱스:", index_info['long_index'])
    print("매도 인덱스:", index_info['exit_index'])

if __name__ == "__main__":
    main()