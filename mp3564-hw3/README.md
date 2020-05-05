The main script you will want to make use of is in evaluate.py. You can either
call it from the command line or import evaluate_models() and use that in your
own code. (There are no requirements on which way you choose to use it, but
you may find it convenient to call it from the command line when testing your
training pipeline or examining specific settings' results for analysis; then
you may want to call the function when evaluating your entire collection of
saved models and printing your table.)

process.py contains helper functions for evaluate.py.

You should not modify either of the provided code files.


The models I saved are named "modelswv.kv" and "svd_wv.txt". 


External resources I used: 

https://stackoverflow.com/questions/31523575/get-u-sigma-v-matrix-from-truncated-svd-in-scikit-learn
