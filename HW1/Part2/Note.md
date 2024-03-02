# Note 
## Written by KYLiN 

---

Part 2:

get page data ->
```json
// same url same title mean image is show in same place
{date:"2023-01-01" , url: "https://example.com" , title:"example" , image_link : "https://image1.png" , hotNumber:0}
{date:"2023-01-01" , url: "https://example.com" , title:"example" , image_link : "https://image2.png" , hotNumber:1}
{date:"2023-01-01" , url: "https://example.com" , title:"example" , image_link : "https://image3.png" , hotNumber:35}
{date:"2023-01-01" , url: "https://example.com" , title:"example" , image_link : "https://image4.png" , hotNumber:100}
...
```

get page image (download)(crawler)
|Date|url|title|image_link|hotNumber|type|download_path|state|
|---| ---|---|---|---|---|---|---|
|2023-01-01| https://example.com|example|https://image1.png|0|0|./image1.png|"OK"|
|2023-01-01| https://example.com|example|https://image1.png|1|0|./image2.png|"OK"|
|2023-01-01| https://example.com|example|https://image1.png|35|1|./image3.png|"Not found"|
|2023-01-01|https://example.com|example|https://image1.png|100|1|./image4.png|"Human"|

> maybe have have human table , need human get the image

get a model to tran (cpu)(image->type)(15mins)(100 image)


CNN Model :
Note : https://zhuanlan.zhihu.com/p/82038049
Pytorch : https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

GPT Recommend : 
```markdown
针对您的需求，您可以考虑使用一些轻量级的CNN模型，这些模型适合于二分类任务，并且可以在CPU上快速运行。以下是一些您可以考虑的PyTorch模型：

LeNet-5: 是最早期的CNN模型之一，适用于轻量级图像分类任务。

MobileNet: 这是由Google开发的一种轻量级卷积神经网络，特别适用于移动和嵌入式设备。您可以选择MobileNetV1或MobileNetV2。

ShuffleNet: 这是另一个由微软提出的轻量级CNN模型，其结构更简单，适用于资源受限的场景。

SqueezeNet: 这是一种具有高效参数的轻量级CNN模型，适用于资源受限的环境。

ResNet-18: 虽然ResNet-18相对较大，但仍然是一种有效的轻量级模型，具有较高的性能。

在Sklearn中，由于其主要专注于传统机器学习算法，可能没有专门用于图像分类的轻量级模型。但是，您可以尝试使用一些经典的机器学习算法，如支持向量机（SVM）或随机森林（Random Forest），在适当的特征提取和预处理之后，它们也可以用于图像分类任务。

对于您的需求，我建议您首先尝试LeNet-5或MobileNet，这些模型都比较简单，适用于二分类任务，并且可以在CPU上快速运行。您可以在PyTorch的torchvision.models模块中找到这些模型的实现。
```
