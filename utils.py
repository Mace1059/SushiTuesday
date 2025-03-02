import pandas as pd
import pandas_market_calendars as mcal
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Takes initial year and adds num_years to it
def year(initial_year, num_years):
    date_obj = initial_year
    if type(initial_year) == str:
      date_obj = datetime.strptime(initial_year, "%Y-%m-%d")
    new_date = date_obj + relativedelta(years=num_years)
    return new_date, new_date.strftime("%Y-%m-%d")

def day(initial_year, num_days):
    date_obj = initial_year
    if type(initial_year) == str:
      date_obj = datetime.strptime(initial_year, "%Y-%m-%d")
    new_date = date_obj + relativedelta(days=num_days)
    return new_date, new_date.strftime("%Y-%m-%d")


def get_trading_days(start_date, end_date, market='NYSE'):
  calendar = mcal.get_calendar(market)
  schedule = calendar.schedule(start_date=start_date, end_date=end_date)
  return len(schedule)

def get_trading_start_end_data(start_date, end_date, data: pd.DataFrame):
  # print("NUMBER OF TRADING DAYS:", get_trading_days(start_date, end_date))
  # full_range = pd.date_range(start=start_date, end=end_date, freq='B')
  
  # # Reindex the DataFrame to the full date range.
  # # This will insert NaNs for missing trading days.
  # data_full = data.reindex(full_range)
  
  # # Fill missing values using forward fill (i.e., copy the previous day's value).
  # data = data_full.fillna(method='ffill')
  counter = 0
  # print("COLUMNS:", data.index)
  while (start_date not in data.index and counter < 100):
    # print((start_date in data.index))
    _, start_date = day(start_date, 1)
    counter += 1
  
  while (start_date not in data.index and counter < 100):
    # print((end_date in data.index))
    _, start_date = day(start_date, 1)
    counter += 1
  
  if counter > 100: return "CANNOT GET DATA: INVALID DATE"

  return data.loc[start_date:end_date, :]
