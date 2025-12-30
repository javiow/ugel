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
    dabt.daa_visualize(result_portval_dict, top_10_portval_dict)
    dabt.save_daa_outputs(top_10_portval_dict, result_portval_dict)