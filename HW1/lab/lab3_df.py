import pandas as pd

# Example data
data = [
    {
        "like_boo_type": "like",
        "user_name": "Alice",
        "body": "comment1",
        "ip": "1.2.3.4",
        "date": "2024-02-27",
        "time": "12:00:00",
        "cnt": 1,
    },
    {
        "like_boo_type": "boo",
        "user_name": "Bob",
        "body": "comment2",
        "ip": "2.3.4.5",
        "date": "2024-02-28",
        "time": "13:00:00",
        "cnt": 1,
    },
    {
        "like_boo_type": "like",
        "user_name": "Alice",
        "body": "comment3",
        "ip": "1.2.3.4",
        "date": "2024-02-28",
        "time": "14:00:00",
        "cnt": 1,
    },
]

# Create DataFrame
df = pd.DataFrame(data)

# Group by "user_name" and aggregate columns
grouped = (
    df.groupby("user_name")
    .agg(
        {
            "like_boo_type": "first",
            "body": lambda x: list(x),
            "ip": "first",
            "date": "first",
            "time": "first",
            "cnt": "sum",
        }
    )
    .reset_index()
)

print(grouped)
