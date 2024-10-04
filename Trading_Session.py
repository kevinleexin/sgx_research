import pandas as pd
from dataclasses import dataclass
from pytz import timezone

"""
Singapore Exchange Nikkei 225 Index Futures
"""


@dataclass
class Nikkei_225_Index_Futures_Trading_Session:
    def __init__(self, trading_date: str):
        self.trading_date = trading_date
        self.timezone = timezone('Asia/Singapore')
        self.T_Session_Pre_Open_Start = pd.Timestamp(self.trading_date + ' 07:15:00.000000', tz=self.timezone)
        self.T_Session_Pre_Open_End = pd.Timestamp(self.trading_date + '07:30:00.000000', tz=self.timezone)

        self.T_Session_Pre_Closing_Start = pd.Timestamp(self.trading_date + '14:25:00.000000', tz=self.timezone)
        self.T_Session_Pre_Closing_End = pd.Timestamp(self.trading_date + '14:30:00.000000', tz=self.timezone)

        self.T1_Session_Pre_Open_Start = pd.Timestamp(self.trading_date + '14:45:00.000000', tz=self.timezone)
        self.T1_Session_Pre_Open_End = pd.Timestamp(self.trading_date + '14:55:00.000000', tz=self.timezone)

        self.T1_Session_Opening_End = pd.Timestamp(self.trading_date + '17:15:00.000000', tz=self.timezone)

        print(f"Trading Date: {self.trading_date}, T_Session_Pre_Open_Start: {self.T_Session_Pre_Open_Start}, "
              f"T_Session_Pre_Open_End: {self.T_Session_Pre_Open_End}, "
              f"T_Session_Pre_Closing_Start: {self.T_Session_Pre_Closing_Start}, "
              f"T_Session_Pre_Closing_End: {self.T_Session_Pre_Closing_End}, "
              f"T1_Session_Pre_Open_Start: {self.T1_Session_Pre_Open_Start}, "
              f"T1_Session_Pre_Open_End: {self.T1_Session_Pre_Open_End}, "
              f"T1_Session_Opening_End: {self.T1_Session_Opening_End}")

    def is_t_session_pre_open_range(self, ts: pd.Timestamp) -> bool:
        if ts <= self.T_Session_Pre_Open_End:
            return True
        else:
            return False

    def is_t_session_pre_closing_range(self, ts: pd.Timestamp) -> bool:
        if self.T_Session_Pre_Closing_Start <= ts <= self.T_Session_Pre_Closing_End:
            return True
        else:
            return False

    def is_t1_session_pre_open_range(self, ts: pd.Timestamp) -> bool:
        if self.T1_Session_Pre_Open_Start <= ts <= self.T1_Session_Pre_Open_End:
            return True
        else:
            return False

    def reset_trading_date(self, trading_date: str):
        self.trading_date = trading_date
        self.T_Session_Pre_Open_Start = pd.Timestamp(self.trading_date + ' 07:15:00.000000', tz=self.timezone)
        self.T_Session_Pre_Open_End = pd.Timestamp(self.trading_date + '07:30:00.000000', tz=self.timezone)

        self.T_Session_Pre_Closing_Start = pd.Timestamp(self.trading_date + '14:25:00.000000', tz=self.timezone)
        self.T_Session_Pre_Closing_End = pd.Timestamp(self.trading_date + '14:30:00.000000', tz=self.timezone)

        self.T1_Session_Pre_Open_Start = pd.Timestamp(self.trading_date + '14:45:00.000000', tz=self.timezone)
        self.T1_Session_Pre_Open_End = pd.Timestamp(self.trading_date + '14:55:00.000000', tz=self.timezone)

        self.T1_Session_Opening_End = pd.Timestamp(self.trading_date + '05:15:00.000000', tz=self.timezone)

    def get_cur_trading_date(self):
        return self.trading_date

    def is_t_session_opening_range(self, ts: pd.Timestamp) -> bool:
        if self.T_Session_Pre_Open_End < ts < self.T_Session_Pre_Closing_Start:
            return True
        else:
            return False

    def is_t1_session_opening_range(self, ts: pd.Timestamp) -> bool:
        if self.T1_Session_Pre_Open_End < ts <= self.T1_Session_Opening_End:
            return True
        else:
            return False
