import pandas as pd
clean = pd.read_parquet('data/clean/flights_2025_01.parquet')
row = clean[(clean.carrier == 'AA') & (clean.flight_number == 10) &
            (clean.origin == 'LAS') &
            (pd.to_datetime(clean.flight_date) == '2025-01-06')]
print(row[['flight_date','sched_dep','dep','sched_arr','arr']].to_string())