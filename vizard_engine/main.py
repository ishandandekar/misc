import matplotlib
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

matplotlib.use("Agg")

df = pl.read_csv("data/1.csv", dtypes={"TotalCharges": pl.String}).with_columns(
    pl.col("TotalCharges").str.replace(pattern=" ", value="0").alias("TotalCharges")
)

# TODO: TARGET DIST
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
churn_response = (
    df.select(pl.col("Churn")).to_series().value_counts(sort=True, parallel=True)
)
ax.bar(
    x=churn_response.select("Churn").to_numpy().squeeze(),
    height=churn_response.select("count").to_numpy().squeeze(),
    color=["#FDB0C0", "#4A0100"],
)
ax.set_title(
    "Proportion of observations of the response variable",
    fontsize=17,
    loc="center",
)
ax.set_xlabel("churn", fontsize=14)
ax.set_ylabel("proportion of observations", fontsize=13)
ax.tick_params(rotation="auto")
plt.savefig("figs/target_dist.png", format="png")
plt.close()

# TODO: CUSTOMER INFO
colors = ["#E94B3C", "#2D2926"]
l1 = ["gender", "SeniorCitizen", "Partner", "Dependents"]
fig = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
for i in range(len(l1)):
    plt.subplot(2, 2, i + 1)
    ax = sns.countplot(
        x=l1[i], data=df.to_pandas(), hue="Churn", palette=colors, edgecolor="black"
    )
    for rect in ax.patches:
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 2,
            rect.get_height(),
            horizontalalignment="center",
            fontsize=11,
        )
    title = l1[i].capitalize() + " vs Churn"
    plt.title(title)
plt.suptitle("Customer information")
plt.savefig("figs/cust_info.png", format="png")
plt.close()

# TODO: NUMERICAL COLS
num_features = ["tenure", "MonthlyCharges", "TotalCharges"]
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(11, 5))
for i in range(len(num_features)):
    plt.subplot(1, 3, i + 1)
    sns.distplot(df[num_features[i]], color=colors[0])
    title = "Distribution : " + num_features[i]
    plt.title(title)
# plt.tight_layout()
plt.savefig("figs/num_cols_dist.png", format="png")
plt.close()

# TODO: 