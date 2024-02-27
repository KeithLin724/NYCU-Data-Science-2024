# sample = "11/10"


# sample_list = "".join(
#     [
#         item if len((item := part_str.strip())) > 1 else f"0{item}"
#         for part_str in sample.split("/")
#     ]
# )


# print(sample_list)

import json

# 示例字典
data = [
    {"name": "John", "age": 30, "city": "New York"},
    {"name": "Alice", "age": 25, "city": "Los Angeles"},
    {"name": "Bob", "age": 35, "city": "Chicago"},
]

# 将字典写入 JSON Lines 文件
with open("data.jsonl", "w") as file:
    for item in data:
        json.dump(item, file)
        file.write("\n")
