# cse842 this explains the data and the code
1. the twitter-120k data contains the tweets we extracted using twitter API, and it's extracted separately, and need to load it to python one by one, for example, to import the 80000 to 120000 tweets, use the following command in python:
           with open('data80000-end', 'rb') as handle:
               f = pickle.load(handle)
2. the bible contains 9 most frequent languages that happened in twitter from the data extracted in step 1, these are the training data, also the udhr file contains the "declaration of human rights" in these nine languages, these are the training data, to load them into the python, remember to load them into txt version.

3. in build_model.py, you can choose either the n gram size, and the training method naive bayes or logistic regression
