import pandas as pd
df = pd.read_parquet('data/output/benchmark_industria.parquet')
assert len(df) >= 16, f'Solo {len(df)} registros'
assert len(df.columns) >= 30, f'Solo {len(df.columns)} columnas'
print(f'PARQUET OK: {len(df)} registros, {len(df.columns)} columnas')
print(f'ISPs: {sorted(df["marca"].unique())}')

import json
nb = json.loads(open('benchmark_industria_notebook.ipynb', encoding='utf-8').read())
assert len(nb['cells']) >= 16, f'Solo {len(nb["cells"])} celdas'
print(f'NOTEBOOK OK: {len(nb["cells"])} celdas')

from pathlib import Path
ga = Path('.github/workflows/daily_benchmark.yml')
ci = Path('.github/workflows/tests.yml')
print(f'daily_benchmark.yml: {"OK" if ga.exists() else "FALTA"}')
print(f'tests.yml: {"OK" if ci.exists() else "FALTA"}')
