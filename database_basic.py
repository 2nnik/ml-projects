import matplotlib.pyplot as plt
import pandas as pd
import numpy

numpy.random.seed(32)
n = 50

days = numpy.arange(1, n + 1)
fruits = numpy.random.choice(["Apple", "Banana", "Orange"], n)
units_sold = numpy.clip(numpy.random.normal(50, 15, n), 1, 100).astype(int)

price_map = {"Apple": 3, "Banana": 1, "Orange": 2}
price_per_unit = numpy.array([price_map[f] for f in fruits])

df = pd.DataFrame({
    "Day": days,
    "Product": fruits,
    "UnitsSold": units_sold,
    "PricePerUnit": price_per_unit
})

df["revenue"] = df[["UnitsSold", "PricePerUnit"]].product(axis=1)
df["high_sales"] = df["UnitsSold"] > 70

missing_indices = numpy.random.choice(df.index, size=5, replace=False)
df.loc[missing_indices, "UnitsSold"] = numpy.nan

df["UnitsSold"] = df["UnitsSold"].fillna(df["UnitsSold"].mean())


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Bar chart - total revenue per product
total_revenue = df.groupby("Product")["revenue"].sum()
ax1.bar(total_revenue.index, total_revenue.values, color=["red", "yellow", "orange"])
ax1.set_title("Total Revenue per Product")
ax1.set_xlabel("Product")
ax1.set_ylabel("Revenue")
ax1.grid(True, linestyle="--", alpha=0.5)

# Line chart - daily revenue
ax2.plot(df["Day"], df["revenue"], color="blue", linewidth=2, marker="o", markersize=4)
ax2.set_title("Daily Revenue Over 50 Days")
ax2.set_xlabel("Day")
ax2.set_ylabel("Revenue")
ax2.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()
