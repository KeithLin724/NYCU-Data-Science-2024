import json

data = {
    "push": {
        "total": 10,
        "top10": [
            {"user_id": "user1", "count": 5},
            {"user_id": "user2", "count": 3},
            {"user_id": "user3", "count": 2},
        ],
    },
    "boo": {
        "total": 5,
        "top10": [{"user_id": "user4", "count": 3}, {"user_id": "user5", "count": 2}],
    },
}

# Write data to file with new lines and indentation
with open("output.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, separators=(",", ": "), ensure_ascii=False)
