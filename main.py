from benchmark import data_load
from strategy import sample_strategy
from backtest import eval

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

    # 전략 수익률 계산
    df_rtn = eval.calculate_strategy_rtn(df_st, index_info)
    print("전략 수익률 계산 완료")
    print(df_rtn[['rtn', 'strategy_rtn', 'cum_rtn', 'cum_strategy_rtn']].head())

    # 드로우다운 계산
    metrics = eval.evaluate_strategy(df_rtn)
    print("전략 평가 완료")
    print("드로우다운 데이터프레임:")
    print(metrics['DD'].head())
    print("최대 드로우다운 시리즈:")
    print(metrics['MDD'].head())
    print("드로우다운 기간 정보:")
    print(metrics['Longest DD Period'].head())
    print("연평균 수익률 (CAGR):")
    print(metrics['CAGR'].head())
    print("샤프 비율:")
    print(metrics['Sharpe Ratio'].head())




if __name__ == "__main__":
    main()