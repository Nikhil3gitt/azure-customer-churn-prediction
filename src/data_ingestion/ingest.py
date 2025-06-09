"""Data ingestion script for Azure Data Factory replacement when running locally."""

import argparse
import pandas as pd
from sqlalchemy import create_engine

def load_csvs_to_sql(csv_paths, conn_str, table_name):
    engine = create_engine(conn_str)
    for csv in csv_paths:
        df = pd.read_csv(csv)
        df.to_sql(table_name, con=engine, if_exists='append', index=False)
        print(f"Loaded {csv} -> {table_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csvs', nargs='+', required=True, help='List of CSV paths')
    parser.add_argument('--conn', required=True, help='Azure SQL connection string')
    parser.add_argument('--table', default='churn_raw', help='Destination table')
    args = parser.parse_args()
    load_csvs_to_sql(args.csvs, args.conn, args.table)
