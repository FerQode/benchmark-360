"""Quick inspection of existing Parquet files."""
import pandas as pd
from pathlib import Path

for p in sorted(Path("data/output").glob("*.parquet")):
    try:
        df = pd.read_parquet(p)
        print(f"\n{p.name}: {len(df)} rows, {len(df.columns)} cols")
        if len(df) > 0:
            print(f"  ISPs: {df['marca'].unique().tolist()}")
            print(f"  Cols: {list(df.columns)}")
            print(df[["marca", "nombre_plan", "velocidad_download_mbps", "precio_plan"]].head(5).to_string())
    except Exception as e:
        print(f"{p.name}: ERROR {e}")
