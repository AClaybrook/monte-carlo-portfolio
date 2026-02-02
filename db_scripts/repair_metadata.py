"""
Repair script to rebuild ticker metadata from existing stock_prices data.

This fixes the issue where metadata intervals are out of sync with actual data,
causing the DataManager to repeatedly try downloading data that already exists.
"""

import sqlite3
import json
from datetime import datetime, timedelta


def get_data_intervals(cursor, ticker: str, max_gap_days: int = 7) -> list:
    """
    Analyze existing data and create proper intervals.
    Groups dates into intervals, allowing for weekend/holiday gaps.
    """
    cursor.execute('''
        SELECT date FROM stock_prices
        WHERE ticker = ?
        ORDER BY date
    ''', (ticker,))

    dates = [row[0] for row in cursor.fetchall()]
    if not dates:
        return []

    # Convert string dates to date objects for comparison
    from datetime import date as dt_date
    parsed_dates = []
    for d in dates:
        if isinstance(d, str):
            parsed_dates.append(dt_date.fromisoformat(d))
        else:
            parsed_dates.append(d)

    # Build intervals
    intervals = []
    current_start = parsed_dates[0]
    current_end = parsed_dates[0]

    for d in parsed_dates[1:]:
        gap = (d - current_end).days
        if gap <= max_gap_days:
            current_end = d
        else:
            intervals.append({
                'start': current_start.isoformat(),
                'end': current_end.isoformat()
            })
            current_start = d
            current_end = d

    # Don't forget the last interval
    intervals.append({
        'start': current_start.isoformat(),
        'end': current_end.isoformat()
    })

    return intervals


def repair_metadata(db_path: str = 'stock_data.db', dry_run: bool = False):
    """
    Rebuild ticker_metadata from actual stock_prices data.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all tickers with data
    cursor.execute('SELECT DISTINCT ticker FROM stock_prices ORDER BY ticker')
    tickers = [row[0] for row in cursor.fetchall()]

    print(f"Found {len(tickers)} tickers with price data")
    print("=" * 60)

    fixed_count = 0
    created_count = 0

    for ticker in tickers:
        # Get actual data bounds
        cursor.execute('''
            SELECT MIN(date), MAX(date), COUNT(*)
            FROM stock_prices WHERE ticker = ?
        ''', (ticker,))
        first_date, last_date, row_count = cursor.fetchone()

        # Check existing metadata
        cursor.execute('''
            SELECT first_valid_date, last_valid_date, data_intervals_json
            FROM ticker_metadata WHERE ticker = ?
        ''', (ticker,))
        existing = cursor.fetchone()

        # Calculate proper intervals
        intervals = get_data_intervals(cursor, ticker)
        intervals_json = json.dumps(intervals)

        if existing:
            existing_first, existing_last, existing_intervals = existing

            # Check if metadata needs updating
            needs_update = (
                existing_first != first_date or
                existing_last != last_date or
                existing_intervals != intervals_json
            )

            if needs_update:
                print(f"FIXING {ticker}:")
                print(f"  Old: {existing_first} to {existing_last}")
                print(f"  New: {first_date} to {last_date} ({row_count} rows, {len(intervals)} interval(s))")

                if not dry_run:
                    cursor.execute('''
                        UPDATE ticker_metadata
                        SET first_valid_date = ?,
                            last_valid_date = ?,
                            data_intervals_json = ?,
                            last_updated = ?
                        WHERE ticker = ?
                    ''', (first_date, last_date, intervals_json, datetime.now(), ticker))

                fixed_count += 1
            else:
                print(f"OK     {ticker}: {first_date} to {last_date} ({row_count} rows)")
        else:
            print(f"CREATE {ticker}:")
            print(f"  {first_date} to {last_date} ({row_count} rows, {len(intervals)} interval(s))")

            if not dry_run:
                cursor.execute('''
                    INSERT INTO ticker_metadata
                    (ticker, first_valid_date, last_valid_date, data_intervals_json, last_updated, is_valid)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (ticker, first_date, last_date, intervals_json, datetime.now(), True))

            created_count += 1

    if not dry_run:
        conn.commit()

    conn.close()

    print("=" * 60)
    print(f"Summary: {created_count} created, {fixed_count} fixed")
    if dry_run:
        print("(DRY RUN - no changes made)")
    else:
        print("Metadata repair complete!")


if __name__ == '__main__':
    import sys

    dry_run = '--dry-run' in sys.argv

    if dry_run:
        print("DRY RUN MODE - showing what would be changed\n")

    repair_metadata(dry_run=dry_run)
