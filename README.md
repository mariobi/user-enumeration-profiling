# User Enumeration Based Profiling

Anonymized dataset and code accompanying the paper

> M. Bischof and E. Portmann, *Towards Intelligent User Enumeration Based Profiling.* In: Innovative
> Security Solutions for Information Technology and Communications (SecITC 2024), Lecture Notes in
> Computer Science, Springer, Cham, 2025, pp. 57–69. https://doi.org/10.1007/978-3-031-87760-5_5

User enumeration means to find out, if a user (e.g., an email address) is registered on a web
application. In the paper, we performed user enumeration on 111 web applications for 396 Swiss test
subjects and explored if the resulting registration pattern can be used to predict attributes of a
person. This repository provides the anonymized user enumeration data, our mean-based membership
algorithm (Algorithm 1 of the paper), the comparison with 9 classical machine learning algorithms
for gender identification and a generic stub which illustrates the user enumeration technique.

## Dataset

`Data.xlsx` holds one row per subject (396 rows) and 112 columns:

| Column | Values |
|---|---|
| `gender` | `0` = male (225 subjects), `1` = female (171 subjects) |
| `A1` … `M3` | one column per web application (111 in total): `1` = registered, `0` = not registered |

The data is strictly anonymized, i.e., it does not contain any names, email addresses or other
personal data. Further, the web applications are replaced by neutral identifiers, so no platform
names are disclosed.

## Contents

| Folder | Purpose |
|---|---|
| `mean_based_membership/` | Mean-based membership algorithm (Algorithm 1 of the paper) and its saved outputs, including the age analysis (`ages.txt`, `ages_means.txt`). |
| `ANN/` | Artificial neural network (Keras, via scikeras). |
| `decision_tree/` | Decision tree, with an analysis of the maximum depth. |
| `gaussian/` | Gaussian naive Bayes, with parameter search. |
| `knn/` | k-nearest neighbors, with parameter search. |
| `logisticregression/` | Logistic regression, with parameter search. |
| `perceptron/` | Perceptron, with parameter and iteration search. |
| `random_forest/` | Random forest, with parameter search. |
| `SGD/` | Stochastic gradient descent classifier, with parameter search. |
| `SVC/` | Support vector classifier, with parameter search. |
| `AccountGrabberStub.example` | Generic stub of the enumeration technique (see below). |

The `*_avg.py` and base scripts are used to train and evaluate a classifier, while the
`*_param*search.py` scripts search for its hyperparameters. The `.txt` files resemble the saved
console outputs of the runs reported in the paper. Since `Data.xlsx` does not contain ages, the
outputs of the age analysis are kept as recorded.

## AccountGrabber stub

`AccountGrabberStub.example` illustrates the two user enumeration methods of the AccountGrabber tool
described in the paper, which are a plain HTTP check (a session, a CSRF token and a validation
request whose response leaks if an email address is registered) and a login check using Selenium.
All URLs, field names and response messages are placeholders. Due to ethical concerns, we abstained
from publishing the actual tool and its platform-specific modules, so the repository can not be
abused against real services. The stub serves as an illustration only and does not run as is.

## Requirements

- Python 3.9 or later
- pandas, openpyxl, numpy, scipy, scikit-learn, imbalanced-learn, matplotlib, seaborn
- tensorflow/keras and scikeras, for the ANN only
- parfit, for the SGD parameter search only

```bash
pip install pandas openpyxl numpy scipy scikit-learn imbalanced-learn matplotlib seaborn
pip install tensorflow scikeras parfit
```

## Usage

Each script reads the dataset from `../Data.xlsx`, therefore it has to be run from inside its own folder:

```bash
cd mean_based_membership
python mean_based_membership.py
```

```bash
cd random_forest
python random_forest_avg.py
```

## Ethics

All procedures used for the user enumeration ensured that the owner of an email address is not
accidentally informed and that no account creations or lockouts did occur. This repository is
intended for research on user enumeration and its mitigation, the paper explains how web
applications can prevent it.

## Citation

```bibtex
@inproceedings{bischof2025userenum,
  author    = {Bischof, Mario and Portmann, Edy},
  title     = {Towards Intelligent User Enumeration Based Profiling},
  booktitle = {Innovative Security Solutions for Information Technology and Communications (SecITC 2024)},
  editor    = {Morogan, Luciana and Roenne, Peter and Bica, Ion},
  series    = {Lecture Notes in Computer Science},
  publisher = {Springer},
  address   = {Cham},
  year      = {2025},
  pages     = {57--69},
  doi       = {10.1007/978-3-031-87760-5_5}
}
```
