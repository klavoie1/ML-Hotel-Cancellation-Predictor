"""
Unit tests for data cleaning and password security.

Run from the project folder:
    python -m pytest
"""
import pandas as pd

import security
from train_model import CATEGORICAL, NUMERIC, clean


def sample_bookings():
    row = {"hotel": "City Hotel", "is_canceled": 0, "lead_time": 10, "arrival_date_year": 2016,
           "arrival_date_month": "July", "arrival_date_week_number": 27,
           "arrival_date_day_of_month": 1, "stays_in_weekend_nights": 1,
           "stays_in_week_nights": 2, "adults": 2, "children": 0, "babies": 0, "meal": "BB",
           "country": "PRT", "market_segment": "Online TA", "distribution_channel": "TA/TO",
           "is_repeated_guest": 0, "previous_cancellations": 0,
           "previous_bookings_not_canceled": 0, "reserved_room_type": "A",
           "assigned_room_type": "A", "booking_changes": 0, "deposit_type": "No Deposit",
           "agent": 9.0, "company": None, "days_in_waiting_list": 0,
           "customer_type": "Transient", "adr": 100.0, "required_car_parking_spaces": 0,
           "total_of_special_requests": 1, "reservation_status": "Check-Out",
           "reservation_status_date": "2016-07-04"}

    rows = [dict(row) for _ in range(5)]
    rows[1].update(children=None, agent=None, meal="Undefined")  # missing values
    rows[2].update(adults=0)                                     # zero guests
    rows[3].update(adr=-5.0)                                     # negative price
    rows[4].update(adr=5400.0)                                   # extreme price
    return pd.DataFrame(rows)


def test_leakage_and_fairness_columns_removed():
    cleaned = clean(sample_bookings())
    for column in ["reservation_status", "reservation_status_date", "country"]:
        assert column not in cleaned.columns

# zero-guest and negative-price rows dropped
def test_invalid_rows_removed():
    assert len(clean(sample_bookings())) == 3


def test_missing_values_handled():
    cleaned = clean(sample_bookings())
    assert cleaned[NUMERIC + CATEGORICAL].isna().sum().sum() == 0
    assert set(cleaned["has_agent"]) == {0, 1}
    assert "Undefined" not in set(cleaned["meal"])


def test_extreme_price_capped():
    assert clean(sample_bookings())["adr"].max() == 500


def test_arrival_date_built():
    assert str(clean(sample_bookings())["arrival_date"].iloc[0].date()) == "2016-07-01"


def test_user_accounts_in_database(tmp_path, monkeypatch):
    import database
    monkeypatch.setattr(database, "DB_FILE", tmp_path / "test.db")
    security.add_user("tester", "Long-Password-9")
    assert security.check_login("tester", "Long-Password-9")
    assert not security.check_login("tester", "wrong-password")


def test_password_hashing():
    record = security.hash_password("Correct-Horse-1")
    assert "Correct-Horse-1" not in record.values()
    assert security.verify_password("Correct-Horse-1", record)
    assert not security.verify_password("wrong-password", record)
