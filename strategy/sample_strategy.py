import numpy as np
import pandas as pd

def sample_strategy(data: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """
    이동평균선 골든크로스/데드크로스 전략 예시
    :param data: 주가 데이터
    :param params: {'short': 20, 'long': 60}
    :return: 매수/매도 신호가 추가된 데이터프레임
    """
    params = params or {'short': 20, 'long': 60}
    df = data.copy()

    df['ma_short'] = df['Close'].rolling(params['short']).mean()
    df['ma_long']  = df['Close'].rolling(params['long']).mean()
    df['signal'] = 0

    df.loc[:, 'signal'] = np.where(df['ma_short'] >= df['ma_long'], 1, 0)

    return df

def get_short_long_index(data: pd.DataFrame) -> dict:
    """
    매수/매도 인덱스  추출 함수
    :param data: 매수/매도 신호가 포함된 주가 데이터프레임
    :return: 매수/매도 신호가 포함된 딕셔너리
    """

    result = {
        'exit_index': [],
        'long_index': []
    }

    if( 'signal' not in data.columns):
        raise ValueError("데이터에 'signal' 컬럼이 없습니다.")

    exit_index = data[(data['signal'] - data['signal'].shift()) == -1].index
    long_index = data[(data['signal'] - data['signal'].shift()) == 1].index

    result['exit_index'] = exit_index
    result['long_index'] = long_index

    return result