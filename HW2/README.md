# HW2 Prompt engineering
## Written BY KYLiN

--- 

Question : https://www.kaggle.com/competitions/data-science-hw2-prompt-engineering/overview

---
### Setup
Make sure you have a api key in `.env` file 
> Web: https://console.groq.com/keys


**Format like**
```sh
# .env file
GROQ_API_KEY=${API_KEY}
```
---
### Package
run this pip command
```sh
pip install -r ./requirements.txt
```

---
### How to use 
`--help` : List the function about the app 

`-f` : Input the csv file ([Example](./data-science-hw2-prompt-engineering/submit.csv))

`-id` : Select the model you want to use [1: llama2 ,2: mixtral, 3: gemma] ([More Detail](https://console.groq.com/docs/models))

Output file is `{model}_pre.csv` and `{model}_ans.csv`, the `{model}_pre.csv` is original response, and ans is take the answer in the response.