# Methods

Package which implements various methods to predict top 10 recipients of emails.

## [baseline.py](baseline.py)

Baseline method predicts always the 10 most frequent recipients of the given sender. 
The result might differ slightly from that of `baseline.py` provided on `Kaggle` 
because of recipients with the same frequency. 
If the 9th, 10th and 11th most frequent recipient have the same frequency, 
two of them might be picked by `baseline.py` while another two of them picked by our baseline method.