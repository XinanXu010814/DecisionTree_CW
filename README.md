# Decision_Tree_CW

## Structure of our project

```sh
.
├── README.md
├── decision_tree.py       # core algorithm for training the decision tree
├── main.py                # main python file, run it to accuracy feedback
│                          # before and after pruning
├── metrics.py             # run it to get more metrics information
├── requirements.txt
└── wifi_db
    ├── clean_dataset.txt
    └── noisy_dataset.txt
```

## How to run our code

In our project folder, simply run:

```sh
python main.py
```

Which will show you something like:

```sh
$ python main.py
Enter the path of the data set:

```

Now input the path to one of the dataset files, and hit enter:

```sh
$ python main.py
Enter the path of the data set:
./wifi_db/clean_dataset.txt
```

or

```sh
./wifi_db/noisy_dataset.txt
```

or **you may enter without any input to run the default files "clean_dataset.txt"
and "noisy_dataset.txt".**

Then the program should be running now. And generate output of the format
below. By the end of training a file,

(The example entered directly for the first input request)

```sh
$ python main.py 
Enter the path of the data set: 


Training ./wifi_db/clean_dataset.txt
===Pruning Off===
Mean accuracy is: 0.973
Mean depth is: 12.3
===Pruning On===
Mean accuracy is: 0.9663
Mean depth is: 8.3

Training ./wifi_db/noisy_dataset.txt
===Pruning Off===
Mean accuracy is: 0.7955
Mean depth is: 19.3
===Pruning On===
Mean accuracy is: 0.8807
Mean depth is: 13.9
```

To replicate our metrics result used in report, you can run the following command:

```sh
$ python metrics.py 

===== Pruning off =====
===Clean Dataset===
Mean accuracy is:
 0.9730000000000001
Mean confusion is:
 [[49.4  0.   0.4  0.2]
 [ 0.  48.1  1.9  0. ]
 [ 0.3  1.8 47.8  0.1]
 [ 0.5  0.   0.2 49.3]]
Mean precision is:
 [0.98457891 0.96381571 0.95176961 0.99395077]
Mean recall is:
 [0.98701448 0.96291219 0.95872755 0.98617087]
Mean f1 is:
 [0.98559528 0.96306533 0.95478675 0.9899227 ]
===Noisy Dataset===
Mean accuracy is:
 0.7955
Mean confusion is:
 [[37.5  3.4  3.9  4.2]
 [ 2.7 40.4  3.8  2.8]
 [ 2.6  3.8 41.7  3.4]
 [ 3.8  2.8  3.7 39.5]]
Mean precision is:
 [0.80792505 0.80100618 0.78899638 0.79245037]
Mean recall is:
 [0.76655515 0.81147761 0.80757466 0.79328359]
Mean f1 is:
 [0.78519435 0.80519198 0.79657979 0.79203292]

===== Pruning on =====
===Clean Dataset===
Mean accuracy is:
 0.9663333333333333
Mean confusion is:
 [[49.61111111  0.          0.28888889  0.1       ]
 [ 0.         47.93333333  2.06666667  0.        ]
 [ 0.6         2.5        46.62222222  0.27777778]
 [ 0.46666667  0.          0.43333333 49.1       ]]
Mean precision is:
 [0.97988437 0.95151636 0.94521577 0.99250713]
Mean recall is:
 [0.99200133 0.95921438 0.93359393 0.98215921]
Mean f1 is:
 [0.98572531 0.95482932 0.93843838 0.98713463]
===Noisy Dataset===
Mean accuracy is:
 0.8806666666666666
Mean confusion is:
 [[44.16666667  1.15555556  1.51111111  2.16666667]
 [ 2.         43.91111111  2.55555556  1.23333333]
 [ 2.2         3.34444444 44.12222222  1.83333333]
 [ 2.34444444  1.56666667  1.95555556 43.93333333]]
Mean precision is:
 [0.87189505 0.88103812 0.88056758 0.89263098]
Mean recall is:
 [0.90071948 0.88090235 0.85450737 0.88207257]
Mean f1 is:
 [0.88505076 0.87988268 0.86613942 0.88665012]
```
