#!/usr/bin/env python
"""
Database Migration Script

Adds new columns to ticker_metadata table for interval tracking.
Safe to run multiple times - checks if columns exist first.
"""

import sqlite3
import sys
from pathlib import Path


def migrate_database(db_path: str = 'stock_data.db'):
    """Add missing columns to ticker_metadata table"""

    if not Path(db_path).exists():
        print(f"Database not found: {db_path}")
        print("No migration needed - new database will be created with correct schema.")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check existing columns
    cursor.execute("PRAGMA table_info(ticker_metadata)")
    existing_columns = {row[1] for row in cursor.fetchall()}

    print(f"Existing columns in ticker_metadata: {existing_columns}")

    migrations = [
        ("last_valid_date", "DATE"),
        ("is_valid", "BOOLEAN DEFAULT 1"),
        ("data_intervals_json", "TEXT"),
    ]

    for col_name, col_type in migrations:
        if col_name not in existing_columns:
            try:
                sql = f"ALTER TABLE ticker_metadata ADD COLUMN {col_name} {col_type}"
                cursor.execute(sql)
                print(f"✓ Added column: {col_name}")
            except sqlite3.OperationalError as e:
                print(f"⚠ Could not add {col_name}: {e}")
        else:
            print(f"  Column already exists: {col_name}")

    conn.commit()
    conn.close()
    print("\n✓ Migration complete!")


if __name__ == '__main__':
    db_path = sys.argv[1] if len(sys.argv) > 1 else 'stock_data.db'
    migrate_database(db_path)