import random
from lib import daa_lib as dabt
from daa_data_load import load_df_dict_pickle

if __name__ == "__main__":

    ticker_df = load_df_dict_pickle("./data/daa/daa_etf_prices.pkl")

    if ticker_df is None or ticker_df.empty:
        raise ValueError("DAA 데이터셋을 불러오지 못했습니다. daa_data_load.py를 먼저 실행하여 데이터를 생성하세요.")

    if len(ticker_df.columns) < 7:
        raise ValueError("DAA 데이터셋에 충분한 티커가 없습니다. daa_data_load.py를 다시 실행하여 더 많은 데이터를 생성하세요.")

    all_ticker_columns = random.sample(ticker_df.drop(columns=["QQQ", "IEF"]).columns.tolist(), 5)
    all_ticker_columns.append("QQQ")
    all_ticker_columns.append("IEF")

    result_portval_dict, top_10_portval_dict = dabt.daa_backtest(ticker_df, all_ticker_columns,  n_samples=100, random_state=None)

    # 시각화는 노트북에서 확인
    # dabt.daa_visualize(result_portval_dict, top_10_portval_dict)

    # 조건에 만족하는 포트폴리오만 저장
    dabt.save_daa_outputs(result_portval_dict, min_sharpe=0.70, min_cagr=0.13, min_mdd=-0.50)