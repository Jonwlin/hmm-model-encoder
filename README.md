## hmm-model-encoder

To run this code:
1. git clone to local repository

Train the model on en_train_tagged.txt which generates a model file with all relavent probabilities and counts
2. Run all the cells from top to bottom, last cell prints the trigram and bigram counts as well as writes to model

Using the model, decode and tag the POS in the raw untagged speech txt file
3. python run.py test en_dev_raw.txt
** this is currently WIP **

Cleaning the folder of miscellaneous models
1. python run.py clean
