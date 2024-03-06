# %%
from ModelHW import ModelHW, ImageDataset
import torchvision.transforms as transforms

# %%
model_hw = ModelHW.load_model("./model/mobileNet_v3_v8_28_BOTH.pth")
model_hw.model

# %%
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
# %%
import os

image_lists = ["./data/HOT", "./data/BOO"]

hot_image_folder, boo_image_folder = (
    os.listdir(image_lists[0]),
    os.listdir(image_lists[1]),
)

hot_images, boo_images = (
    [os.path.join(image_lists[0], item) for item in hot_image_folder],
    [os.path.join(image_lists[1], item) for item in boo_image_folder],
)

# %%
test_images = hot_images[:10] + boo_images[:10]

dataset = ImageDataset(test_images, transform)
dataset


# %%
from torch.utils.data import DataLoader

data_loader = DataLoader(dataset, batch_size=5)

# %%
all_output = []
for item in data_loader:
    output = model_hw.prediction(item)
    all_output.extend(output.tolist())

# %%
all_output

# %%
all_output = [int(number == 3) for number in all_output]

# %%
all_output
# %%
true_table = [1] * 10 + [0] * 10

# %%
from sklearn.metrics import f1_score

f1 = f1_score(true_table, all_output)

# %%
f1

# %%
