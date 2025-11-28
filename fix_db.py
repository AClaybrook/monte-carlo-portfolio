from data_manager import DataManager, TickerMetadata
from sqlalchemy import text

def reset_metadata():
    dm = DataManager()

    print("⚠ Wiping corrupted Ticker Metadata...")
    try:
        dm.session.query(TickerMetadata).delete()
        dm.session.commit()
        print("✓ Ticker Metadata table cleared.")
    except Exception as e:
        dm.session.rollback()
        print(f"Error: {e}")
    finally:
        dm.close()

if __name__ == "__main__":
    reset_metadata()