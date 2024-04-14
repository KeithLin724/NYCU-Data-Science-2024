# DS HW3 Headline Generation

## Written BY KYLiN

---

### How to run it

#### First

you need put the train and test source in data folder

```sh
# run all 
# inside have 2 step 
## first is train a model 
## second is using the model to generate the result
sh 109511276.sh

# run training 
jupyter-nbconvert --execute --to pdf use_hugging_face.ipynb

# in generate the result
## inside is using check-point 6300 
## you can change the check-point if you find better
jupyter-nbconvert --execute --to pdf summit_predition.ipynb
```

### Result

You will get a `use_hugging_face.pdf`, `summit_predition.pdf` and model after runing the shell script

### Reference

- <https://huggingface.co/blog/how-to-generate>
