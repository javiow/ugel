import numpy as np
import pandas as pd


def get_returns_df(df: pd.DataFrame, N: int = 1, log: bool = False) -> pd.DataFrame:
    """
    주가 데이터프레임에서 수익률을 계산하는 함수
    :param df: 주가 데이터프레임 (종가 컬럼이 있어야 함)
    :param N: 수익률 계산 기간 (기본값: 1일)
    :param log: 로그 수익률 여부 (기본값: False)
    :return: 수익률 데이터프레임
    """

    if log:
        return np.log(df / df.shift(N)).iloc[N - 1:].fillna(0)
    else:
        return df.pct_change(N, fill_method=None).iloc[N - 1:].fillna(0)


def get_cum_returns_df(return_df: pd.DataFrame, log: bool = False) -> pd.DataFrame:
    """
    누적 수익률을 계산하는 함수
    :param return_df: 수익률 데이터프레임
    :param log: 로그 수익률 여부
    :return: 누적 수익률 데이터프레임
    """
    if log:
        return np.exp(return_df.cumsum())
    else:
        return (1 + return_df).cumprod()  # same with (return_df.cumsum() + 1)


def get_CAGR_series(cum_rtn_df: pd.DataFrame, num_day_in_year: int = 250) -> pd.Series:
    """
    연평균 수익률(CAGR)을 계산하는 함수
    :param cum_rtn_df: 누적 수익률 데이터프레임
    :param num_day_in_year: 연간 시장 오픈일수
    :return: 연평균 수익률 시리즈
    """

    cagr_series = cum_rtn_df.iloc[-1] ** (num_day_in_year / (len(cum_rtn_df))) - 1
    return cagr_series


def get_sharpe_ratio(log_rtn_df: pd.DataFrame, yearly_rfr: float = 0.025) -> pd.Series:
    """
    샤프 비율을 계산하는 함수
    :param log_rtn_df: 로그 수익률 데이터프레임
    :param yearly_rfr: 국고채 수익률(고정값)
    :return: 샤프 비율 시리즈
    """
    excess_rtns = log_rtn_df.mean() * 252 - yearly_rfr
    return excess_rtns / (log_rtn_df.std() * np.sqrt(252))


def get_drawdown_infos(cum_returns_df: pd.DataFrame):
    """
    누적 수익률 데이터프레임에서 드로우다운 정보와 최대 드로우다운, 가장 긴 드로우다운 기간을 계산하는 함수
    :param cum_returns_df: 누적 수익률 데이터프레임
    :return dd_df: 드로우다운 데이터프레임,
            mdd_series: 최대 드로우다운 시리즈,
            dd_duration_info_df: 드로우다운 기간 정보 데이터프레임
    """

    # 1. Drawdown 계산
    cummax_df = cum_returns_df.cummax()
    dd_df = cum_returns_df / cummax_df - 1

    # 2. 가장 긴 drawdown
    mdd_series = dd_df.min()

    # 3. 가장 긴 드로우다운 기간 정보 계산
    dd_duration_info_list = list()
    max_point_df = dd_df[dd_df == 0]
    for col in max_point_df:
        _df = max_point_df[col]
        _df.loc[dd_df[col].last_valid_index()] = 0
        _df = _df.dropna()

        periods = _df.index[1:] - _df.index[:-1]

        days = periods.days
        max_idx = days.argmax()

        longest_dd_period = days.max()
        dd_mean = int(np.mean(days))
        dd_std = int(np.std(days))

        dd_duration_info_list.append(
            [
                dd_mean,
                dd_std,
                longest_dd_period,
                "{} ~ {}".format(_df.index[:-1][max_idx].date(), _df.index[1:][max_idx].date())
            ]
        )

    dd_duration_info_df = pd.DataFrame(
        dd_duration_info_list,
        index=dd_df.columns,
        columns=['drawdown mean', 'drawdown std', 'longest days', 'longest period']
    )
    return dd_df, mdd_series, dd_duration_info_df

def calculate_strategy_rtn(data: pd.DataFrame, short_long_index: dict) -> pd.DataFrame:
    """
    Buy & Hold 전략과 비교하기 위한 수익률, 누적수익률 계산 함수
    :param data: 매수/매도 신호가 포함된 주가 데이터프레임
    :param short_long_index: 매수/매도 인덱스 딕셔너리
    :return: 수익률, 전략수익률, 누적수익률, 전략 누적수익률 지표가 추가된 데이터프레임
    """

    if 'signal' not in data.columns:
        raise ValueError("데이터에 'signal' 컬럼이 없습니다.")

    if 'Close' not in data.columns:
        raise ValueError("데이터에 'Close' 컬럼이 없습니다.")

    # buy&hold 수익률, 전략 수익률 계산
    data.loc[:, 'rtn'] = get_returns_df(data['Close'], log=True)
    data.loc[:, 'strategy_rtn'] = (data['signal'].shift(1) * data['rtn']).fillna(0)

    # buy&hold 누적수익률, 전략 누적수익률 계산
    data.loc[:, 'cum_rtn'] = get_cum_returns_df(data['rtn'], log=True)
    data.loc[:, 'cum_strategy_rtn'] = get_cum_returns_df(data['strategy_rtn'], log=True)

    exit_index = short_long_index['exit_index']
    long_index = short_long_index['long_index']

    # buy&hold 전략과 비교
    # 매수: 빨간색점 / 매도: 주황색점
    ax = data[['cum_rtn', 'cum_strategy_rtn']].plot(figsize=(10, 5))
    data.loc[exit_index, "cum_rtn"].plot(ax=ax, marker="o", markersize=8, linestyle="none", color="red")
    data.loc[long_index, "cum_rtn"].plot(ax=ax, marker="o", markersize=8, linestyle="none", color="orange")

    return data

def evaluate_strategy(data: pd.DataFrame) -> dict:
    """
    전략 평가 함수 - Sharpe Ratio, MDD, CAGR
    :param data: 수익률, 누적수익률이 포함된 주가 데이터프레임
    :return dict: 평가 지표가 포함된 딕셔너리
    """

    if 'rtn' not in data.columns or 'strategy_rtn' not in data.columns:
        raise ValueError("데이터에 'rtn' 또는 'strategy_rtn' 컬럼이 없습니다.")

    # 샤프 비율
    sharpe_ratio = get_sharpe_ratio(data[['rtn', 'strategy_rtn']]).to_frame("Sharpe Ratio")

    # MDD 정보 조회: 드로우다운 데이터프레임, 최대 드로우다운 시리즈, 가장 긴 드로우다운 기간 정보 데이터프레임
    dd_df, mdd_series, longest_dd_period_df = get_drawdown_infos(data.set_index('Date').filter(like="cum_"))
    mdd = mdd_series.to_frame("MDD")

    # 드로우다운 시각화
    dd_df.plot(figsize=(10, 5))

    # 연평균 수익률
    cagr = get_CAGR_series(data.filter(like="cum_")).to_frame("CAGR")

    # 평가 지표 계산
    metrics = {
        'Sharpe Ratio': sharpe_ratio,
        'DD': dd_df,
        'MDD': mdd,
        'Longest DD Period': longest_dd_period_df,
        'CAGR': cagr
    }

    return metrics