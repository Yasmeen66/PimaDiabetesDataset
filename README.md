# PimaDiabetesDataset
## Code Documentation: Pima Indians Diabetes Classification

This code aims to build a classification model to predict the presence or absence of diabetes in Pima Indian women based on various medical features. The dataset used contains information about the number of times pregnant, glucose concentration, blood pressure, skinfold thickness, insulin level, body mass index, diabetes pedigree function, and age of the patients.

### 1. Importing Libraries
The necessary libraries are imported in the code to perform various tasks related to data manipulation, visualization, model training, and evaluation. The libraries used are:
- `pandas`: For data handling and manipulation.
- `matplotlib.pyplot`: For data visualization.
- `sklearn` (Scikit-learn): For machine learning algorithms and evaluation metrics.

### 2. Loading the Dataset
The code loads the dataset from the provided URL using the `read_csv` function from `pandas`. The dataset contains information about Pima Indian women and whether they have diabetes or not. The column names are specified in the `names` list.

### 3. Exploratory Data Analysis
Several exploratory data analysis tasks are performed to understand the dataset:

- Printing the shape of the dataset using `dataset.shape` to display the number of rows and columns.
- Displaying the first 20 rows of the dataset using `dataset.head(20)`.
- Generating descriptive statistics of the dataset using `dataset.describe()`, which includes count, mean, standard deviation, minimum, and quartile values for each column.
- Printing the class distribution using `dataset.groupby('Class').size()` to show the number of instances in each class.

### 4. Data Preprocessing
The dataset is preprocessed to prepare it for model training:

- The features (`X`) and the target variable (`y`) are extracted from the dataset. The features consist of columns 0 to 7, while the target variable is in column 8.
- The dataset is split into training and validation sets using `train_test_split` from `sklearn.model_selection`. The validation set will contain 20% of the data, while the training set will contain the remaining 80%.

### 5. Model Evaluation
Various classification algorithms are evaluated using cross-validation:

- A list `models` is created to store the algorithms to be evaluated. The algorithms include Logistic Regression, Linear Discriminant Analysis, K-Nearest Neighbors, Decision Tree Classifier, Gaussian Naive Bayes, and Support Vector Machine.
- The code then performs a cross-validation loop for each algorithm using `StratifiedKFold` with 10 splits. It evaluates the accuracy of each algorithm on the training set using `cross_val_score` and stores the results in `results`. The algorithm names are stored in the `names` list.
- For each algorithm, the code prints the mean accuracy and standard deviation of the cross-validated results.

### 6. Model Training and Prediction
The Support Vector Machine (SVM) algorithm is chosen as the final model based on the evaluation results. The code performs the following steps:

- A new SVM model is instantiated with `'auto'` as the gamma value.
- The model is trained on the training set using `fit` method with `X_train` and `Y_train`.
- Predictions are made on the validation set using the trained model with `predict` method and stored in `predictions`.

### 7. Model Evaluation and Reporting
The predictions made by the SVM model are evaluated and reported:

- The accuracy of the model is calculated by comparing the predicted values (`predictions`) with the actual values (`Y_validation`).
- The confusion matrix is printed using `confusion_matrix` to show the performance of the model in terms of true positive, true negative, false positive, and false negative predictions.
- The classification report is printed using `classification_report` to display precision, recall, F1-score, and support for each class.
