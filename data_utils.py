#!/usr/bin/env python
"""
Data Management Utility

Commands:
    coverage  - Show data coverage report
    gaps      - Find gaps in data for a ticker
    download  - Bulk download tickers
    clear     - Clear data for a ticker
    info      - Show ticker info and inception date
"""

import argparse
from datetime import date, timedelta
from data_manager import DataManager, IntervalTracker
import json


def cmd_coverage(args):
    """Show data coverage report"""
    dm = DataManager(args.db)

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(',')]
    else:
        tickers = dm.list_all_tickers()

    if not tickers:
        print("No tickers found in database.")
        dm.close()
        return

    report = dm.get_data_coverage_report(tickers)

    print("\n" + "="*90)
    print("DATA COVERAGE REPORT")
    print("="*90)

    # Format nicely
    for _, row in report.iterrows():
        ticker = row['ticker']
        first = row['first_date']
        last = row['last_date']
        days = row['days_of_data']
        inception = row['known_inception']
        intervals = row['coverage_intervals']

        status = "✓" if intervals == 1 else f"⚠ {intervals} intervals"

        print(f"\n{ticker}:")
        print(f"  Data range:      {first} to {last}")
        print(f"  Trading days:    {days:,}")
        print(f"  Known inception: {inception or 'Unknown'}")
        print(f"  Coverage:        {status}")

    dm.close()


def cmd_gaps(args):
    """Find gaps in data"""
    dm = DataManager(args.db)

    ticker = args.ticker.upper()
    start = date.fromisoformat(args.start) if args.start else date.today() - timedelta(days=365*10)
    end = date.fromisoformat(args.end) if args.end else date.today()

    gaps = dm.find_gaps(ticker, start, end, max_gap_days=args.max_gap)

    print(f"\n{'='*60}")
    print(f"GAPS IN DATA: {ticker}")
    print(f"Search range: {start} to {end}")
    print(f"Max allowed gap: {args.max_gap} days")
    print(f"{'='*60}")

    if not gaps:
        print("\n✓ No significant gaps found!")
    else:
        print(f"\n⚠ Found {len(gaps)} gap(s):\n")
        for gap_start, gap_end, gap_days in gaps:
            print(f"  {gap_start} → {gap_end} ({gap_days} days)")

    dm.close()


def cmd_download(args):
    """Bulk download tickers"""
    dm = DataManager(args.db)

    tickers = [t.strip().upper() for t in args.tickers.split(',')]
    start = date.fromisoformat(args.start) if args.start else date.today() - timedelta(days=365*10)
    end = date.fromisoformat(args.end) if args.end else date.today()

    print(f"\n{'='*60}")
    print(f"BULK DOWNLOAD")
    print(f"{'='*60}")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Range:   {start} to {end}")
    print(f"Force:   {args.force}")
    print(f"Sequential: {args.sequential}")
    print()

    results = dm.bulk_download(tickers, start, end,
                                force_update=args.force,
                                sequential=args.sequential)

    print(f"\n{'='*60}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*60}")

    for ticker in tickers:
        if ticker in results:
            df = results[ticker]
            print(f"  ✓ {ticker}: {len(df)} rows")
        else:
            print(f"  ✗ {ticker}: No data")

    dm.close()


def cmd_clear(args):
    """Clear data for a ticker"""
    dm = DataManager(args.db)

    ticker = args.ticker.upper()

    if args.yes:
        dm.clear_ticker_data(ticker)
    else:
        confirm = input(f"Clear all data for {ticker}? [y/N]: ")
        if confirm.lower() == 'y':
            dm.clear_ticker_data(ticker)
        else:
            print("Cancelled.")

    dm.close()


def cmd_info(args):
    """Show ticker info"""
    dm = DataManager(args.db)

    ticker = args.ticker.upper()

    print(f"\n{'='*60}")
    print(f"TICKER INFO: {ticker}")
    print(f"{'='*60}")

    # Inception date
    inception = dm.get_ticker_inception_date(ticker)
    print(f"\nKnown inception date: {inception or 'Unknown'}")

    # Interval tracker info
    tracker = dm._get_interval_tracker(ticker)
    bounds = tracker.bounds

    if bounds:
        print(f"Data bounds: {bounds[0]} to {bounds[1]}")
        print(f"\nIntervals:")
        intervals_json = tracker.to_json()
        intervals = json.loads(intervals_json)
        for i, iv in enumerate(intervals, 1):
            print(f"  {i}. {iv['start']} to {iv['end']}")
    else:
        print("No data in database.")

    # Check for gaps
    if bounds:
        gaps = dm.find_gaps(ticker, bounds[0], bounds[1])
        if gaps:
            print(f"\n⚠ Data gaps found: {len(gaps)}")
            for gap_start, gap_end, gap_days in gaps[:5]:
                print(f"  {gap_start} → {gap_end} ({gap_days} days)")
            if len(gaps) > 5:
                print(f"  ... and {len(gaps) - 5} more")
        else:
            print("\n✓ No gaps in data")

    dm.close()


def cmd_list(args):
    """List all tickers in database"""
    dm = DataManager(args.db)

    tickers = sorted(dm.list_all_tickers())

    print(f"\n{'='*60}")
    print(f"TICKERS IN DATABASE ({len(tickers)} total)")
    print(f"{'='*60}\n")

    # Print in columns
    cols = 6
    for i in range(0, len(tickers), cols):
        row = tickers[i:i+cols]
        print("  " + "  ".join(f"{t:10}" for t in row))

    dm.close()


def main():
    parser = argparse.ArgumentParser(description='Data Management Utility')
    parser.add_argument('--db', default='stock_data.db', help='Database path')

    subparsers = parser.add_subparsers(dest='command', help='Command')

    # Coverage command
    p_cov = subparsers.add_parser('coverage', help='Show data coverage report')
    p_cov.add_argument('--tickers', help='Comma-separated tickers (default: all)')

    # Gaps command
    p_gaps = subparsers.add_parser('gaps', help='Find gaps in data')
    p_gaps.add_argument('ticker', help='Ticker symbol')
    p_gaps.add_argument('--start', help='Start date (YYYY-MM-DD)')
    p_gaps.add_argument('--end', help='End date (YYYY-MM-DD)')
    p_gaps.add_argument('--max-gap', type=int, default=7, help='Max allowed gap days')

    # Download command
    p_dl = subparsers.add_parser('download', help='Bulk download tickers')
    p_dl.add_argument('tickers', help='Comma-separated tickers')
    p_dl.add_argument('--start', help='Start date (YYYY-MM-DD)')
    p_dl.add_argument('--end', help='End date (YYYY-MM-DD)')
    p_dl.add_argument('--force', action='store_true', help='Force re-download')
    p_dl.add_argument('--sequential', '-s', action='store_true',
                      help='Download one at a time (slower but avoids rate limits)')

    # Clear command
    p_clr = subparsers.add_parser('clear', help='Clear data for a ticker')
    p_clr.add_argument('ticker', help='Ticker symbol')
    p_clr.add_argument('-y', '--yes', action='store_true', help='Skip confirmation')

    # Info command
    p_info = subparsers.add_parser('info', help='Show ticker info')
    p_info.add_argument('ticker', help='Ticker symbol')

    # List command
    p_list = subparsers.add_parser('list', help='List all tickers in database')

    args = parser.parse_args()

    if args.command == 'coverage':
        cmd_coverage(args)
    elif args.command == 'gaps':
        cmd_gaps(args)
    elif args.command == 'download':
        cmd_download(args)
    elif args.command == 'clear':
        cmd_clear(args)
    elif args.command == 'info':
        cmd_info(args)
    elif args.command == 'list':
        cmd_list(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()