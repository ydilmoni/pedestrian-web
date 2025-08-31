#!/usr/bin/env python3
"""
time_features.py

Provides a helper to attach temporal features to a GeoDataFrame (or DataFrame) based on a single timestamp.
"""
import pandas as pd


def get_time_of_day(hour: int) -> str:
    """
    Bucket hour of day into human-readable periods.

    Returns one of: 'morning', 'afternoon', 'evening', 'night'.
    """
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 21:
        return "evening"
    return "night"


def compute_time_features(gdf, timestamp=None):
    """
    Add Hour, is_weekend, and time_of_day columns to `gdf` based on `timestamp`.

    Parameters
    ----------
    gdf : GeoDataFrame or DataFrame
        Must be mutable; temporal columns will be added in-place.
    timestamp : str or datetime-like, optional
        ISO 8601 string or pandas.Timestamp. Defaults to now.

    Returns
    -------
    GeoDataFrame or DataFrame
        The same `gdf` with new columns:
          - Hour: int [0–23]
          - is_weekend: 0 or 1
          - time_of_day: category ('morning','afternoon','evening','night')
    """
    # parse timestamp
    ts = pd.to_datetime(timestamp) if timestamp is not None else pd.Timestamp.now()

    hour = ts.hour
    weekend_flag = int(ts.weekday() >= 5)
    tod = get_time_of_day(hour)

    gdf = gdf.copy()
    gdf["Hour"] = hour
    gdf["is_weekend"] = weekend_flag
    gdf["time_of_day"] = tod

    return gdf


if __name__ == "__main__":
    # quick smoke test
    import geopandas as gpd
    test = gpd.GeoDataFrame([{}])
    out = compute_time_features(test, "2025-07-18T12:34:00Z")
    print(out)
