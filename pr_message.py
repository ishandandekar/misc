import pandas as pd
from csv2md.table import Table

df = pd.DataFrame({"a": [1, 2, 3], "b": ["hi", "hello", "hey"]})
df.to_csv("sample.csv")

with open("sample.csv") as f:
    table = Table.parse_csv(f)

print(table.markdown())
