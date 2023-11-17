from datetime import datetime, timedelta

def calculate_date(prediction_length):
    base_date = datetime(2019, 10, 30)
    new_date = base_date - timedelta(days=prediction_length * 10)
    return new_date.strftime('%Y-%m-%d')