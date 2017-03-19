#### Description of the code :

* main.py : python script to run the code and try each method


** helpers/ : Package which implements some handy helper functions (IO, *etc*.).

            ## [average_precision.py](average_precision.py)

            Implementation of `Kaggle`'s `precision@10`. Handy for local cross-validation.

            ## [misc.py](misc.py)

            Miscellaneous functions including (but not limited to):

            -   `make_X_y`: reformats `{training,test}_set.csv` and `{train,test}_info.csv`
            into `X_{train,test}` and `y_{train,test}` where `X` contains everything but `recipients`
            and where `y` contains `mid` and `recipients`.
            Note that `y_test` is returned `None` if no `recipients` field is found (the case for *real* test instead of cross-validation).
            -   `training_test_split`: split `training_set.csv` and `training_info.csv` into training and test set.
            It has the same signature to the function in `sklearn`.
            Note that for efficiency, we suppose the input to this function to be raw data instead of the one returned by `make_X_y`.
            into training
            -   `score`: calculates the `precision@10` score
            -  `write_to_file`: writes result to submission file

** methods/ : Package which implements various methods to predict top 10 recipients of emails;


** data/    : Folder to store the data provided by the challenge;

** results/ : Folder to store our submission results obtained by our methods.
