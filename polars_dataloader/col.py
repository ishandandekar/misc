import polars as pl

results = dict()
results["lr"] = [1.0, 2.0, 3.0]
results["pr"] = [4.0, 5.0, 6.0]

df = (
    pl.DataFrame(results)
    .transpose()
    .with_columns(
        pl.col("column_0").alias("train_a"),
        pl.col("column_1").alias("train_b"),
        pl.col("column_2").alias("train_c"),
    )
    .select(pl.col("train_a"), pl.col("train_b"), pl.col("train_c"))
)
print(df)
