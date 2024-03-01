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
