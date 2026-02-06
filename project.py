import pandas as pd 
import matplotlib.pyplot as plt

users = pd.read_csv('users.csv')
sessions = pd.read_csv('sessions.csv')
orders = pd.read_csv('orders.csv')

print(users.shape)
print(sessions.shape)
print(orders.shape)

print(users.head())
print(sessions.head())
print(orders.head())

print(users.dtypes)
print(sessions.dtypes)
print(orders.dtypes)

print(users.isna().sum())
print(sessions.isna().sum())
print(orders.isna().sum())

print(users.duplicated().sum())
print(sessions.duplicated().sum())
print(orders.duplicated().sum())

print(users.isnull().sum())
print(sessions.isnull().sum())
print(orders.isnull().sum())

sessions_stats = (
    sessions
    .groupby('user_id')
    .agg(
        sessions_count=('session_id', 'count'),
        avg_pages=('pages_viewed', 'mean'),
        max_pages=('pages_viewed', 'max'),
    )
    .reset_index()
)

print(sessions_stats.head())

df = (
    users
    .merge(sessions_stats, on =('user_id'), how = 'left')
    .merge(orders, on =('user_id'), how = 'left')
)

print(df.head())
print(df.shape)

df['sessions_stats'] = df['sessions_count'].fillna(0)
df['avg_pages'] = df['avg_pages'].fillna(0)
df['max_pages'] = df['max_pages'].fillna(0)
df['revenue'] = df['revenue'].fillna(0)
df["order_id"] = df["order_id"].fillna(0)
df["order_date"] = df["order_date"].fillna("No Order")
df["made_purchase"] = (df["revenue"] > 0).astype(int)
print(df.head())

conversion = (
    df
    .groupby("ab_group")["made_purchase"]
    .mean()
    .rename("conversion_rate")
)

print(conversion)


arpu = (
    df.
    groupby("ab_group")["revenue"]
    .mean()
    .rename("arpu")
)

print(arpu)

aov = (
    df[df["made_purchase"] == 1]
    .groupby("ab_group")["revenue"]
    .mean()
    .rename("aov")
)
print(aov)

engagement = (
    df
    .groupby("ab_group")[["sessions_count", "avg_pages"]]
    .mean()
)

print(engagement)

metrics = (
    pd.concat([conversion, arpu, aov, engagement], axis=1)
    .round(3)
)

print(metrics)

metrics.plot(kind="bar", figsize = (10, 5))
plt.title("A/B Test Results")
plt.ylabel("Values")
plt.xticks(rotation=0)
plt.show()

df.groupby("ab_group")["revenue"].describe()

group_a = df[df["ab_group"] == "A"]["revenue"]
group_b = df[df["ab_group"] == "B"]["revenue"]
print(group_a.describe())
print(group_b.describe())

print(len(group_a), len(group_b))

from scipy.stats import ttest_ind
t_stat, p_value = ttest_ind(group_a, group_b, equal_var=False)
print(t_stat, p_value)
