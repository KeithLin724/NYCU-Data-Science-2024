cd out

python ./109511276.py crawl
# push
python ./109511276.py push 0101 0130
python ./109511276.py push 0425 0426
python ./109511276.py push 0930 1101

# popular
python ./109511276.py popular 0213 0312
python ./109511276.py popular 0303 0329
python ./109511276.py popular 1002 1105

# keyword
python ./109511276.py keyword 0701 0731 IG
python ./109511276.py keyword 0815 0910 PiTT

cd ../
pwd
python eval.py answer out