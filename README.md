1.data preeproccer


import numpy as np
import pandas as pd

# Read the dataset
dataset = pd.read_csv("Data_preprocessing.csv")

# Check null values
dataset.isnull().sum()

# Dependent and independent variables
x = dataset.iloc[:, 0:3].values
y = dataset.iloc[:, 3:4].values

# Import imputer and fill the null values
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

# Import OneHotEncoder for categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Apply One Hot Encoding to first column
ct = ColumnTransformer(
    transformers=[('onehot', OneHotEncoder(), [0])],
    remainder='passthrough'
)

x = ct.fit_transform(x)

# Avoid dummy variable trap
x = x[:, 1:]

# Split dataset into training and testing sets
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.20,
    random_state=0
)
2.single multiple lineer regression


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv("C:/Users/Admin/OneDrive/Desktop/dataset/GHI_Report.csv")

# Define target and feature columns
target_column = 'H_Score'

simple_feature = ['Economy']
multiple_features = ['Economy', 'Fam', 'Health', 'Freedom']

# Target variable
Y = df[target_column].values

# =========================================================
# SIMPLE LINEAR REGRESSION
# =========================================================

X_simple = df[simple_feature].values

# Create and train model
simple_model = LinearRegression()
simple_model.fit(X_simple, Y)

# Prediction
Y_simple_pred = simple_model.predict(X_simple)

# Evaluation metrics
rmse_simple = np.sqrt(mean_squared_error(Y, Y_simple_pred))
r2_simple = simple_model.score(X_simple, Y)

# Display results
print("\n--- Simple Linear Regression ---")
print("Coefficient (Slope):", simple_model.coef_[0])
print("Intercept:", simple_model.intercept_)
print("RMSE:", rmse_simple)
print("R² Score:", r2_simple)

# Plot Simple Linear Regression
plt.figure()

plt.scatter(
    X_simple,
    Y,
    marker='*',
    label="Actual Data"
)

plt.plot(
    X_simple,
    Y_simple_pred,
    'b-',
    label="Regression Line"
)

plt.xlabel("Economy")
plt.ylabel("H_Score")
plt.title("Simple Linear Regression")

plt.legend()
plt.show()

# =========================================================
# MULTIPLE LINEAR REGRESSION
# =========================================================

X_multi = df[multiple_features].values

# Create and train model
multi_model = LinearRegression()
multi_model.fit(X_multi, Y)

# Prediction
Y_multi_pred = multi_model.predict(X_multi)

# Evaluation metrics
rmse_multi = np.sqrt(mean_squared_error(Y, Y_multi_pred))
r2_multi = multi_model.score(X_multi, Y)

# Display results
print("\n--- Multiple Linear Regression ---")

for feature, coef in zip(multiple_features, multi_model.coef_):
    print(f"Coefficient for {feature}: {coef}")

print("Intercept:", multi_model.intercept_)
print("RMSE:", rmse_multi)
print("R² Score:", r2_multi)

# Plot Multiple Linear Regression
# Actual vs Predicted

plt.figure()

plt.scatter(
    Y,
    Y_multi_pred,
    marker='o'
)

plt.xlabel("Actual H_Score")
plt.ylabel("Predicted H_Score")

plt.title("Multiple Linear Regression: Actual vs Predicted")

# Reference line
plt.plot(
    [Y.min(), Y.max()],
    [Y.min(), Y.max()],
    'r--'
)

plt.show()

print("\nVisualization completed successfully ✅")




3.polynomial regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# =========================================================
# LOAD DATASET
# =========================================================

data = pd.read_csv('GHI_Report.csv')

# Independent and dependent variables
X = data['Economy'].values.reshape(-1, 1)
Y = data['H_Score'].values

# Split dataset into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

# =========================================================
# POLYNOMIAL REGRESSION ANALYSIS
# =========================================================

degrees = [1, 2, 3, 4]
results = []

print('=' * 60)
print('POLYNOMIAL REGRESSION ANALYSIS')
print('=' * 60)

for deg in degrees:

    # Create polynomial features
    poly = PolynomialFeatures(degree=deg)

    # Transform training data
    X_train_poly = poly.fit_transform(X_train)

    # Train model
    model = LinearRegression()
    model.fit(X_train_poly, Y_train)

    # Training score
    train_r2 = model.score(
        poly.transform(X_train),
        Y_train
    )

    # Testing score
    test_r2 = model.score(
        poly.transform(X_test),
        Y_test
    )

    # Predictions
    Y_test_pred = model.predict(
        poly.transform(X_test)
    )

    # RMSE
    test_rmse = np.sqrt(
        mean_squared_error(Y_test, Y_test_pred)
    )

    # Model status
    if train_r2 - test_r2 > 0.15:
        status = 'OVERFITTING'
    elif test_r2 < 0.5:
        status = 'UNDERFITTING'
    else:
        status = 'GOOD FIT'

    # Store results
    results.append({
        'deg': deg,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'rmse': test_rmse,
        'poly': poly,
        'model': model
    })

    # Print results
    print(
        f'Degree {deg}: '
        f'Train R² = {train_r2:.4f}, '
        f'Test R² = {test_r2:.4f}, '
        f'RMSE = {test_rmse:.4f} '
        f'[{status}]'
    )

# Best model
best = max(results, key=lambda x: x['test_r2'])

print(f'\nBest Degree: {best["deg"]} '
      f'(Test R² = {best["test_r2"]:.4f})')

print('=' * 60)

# =========================================================
# PLOT POLYNOMIAL FITS
# =========================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Generate smooth curve values
X_range = np.linspace(
    X.min(),
    X.max(),
    300
).reshape(-1, 1)

for idx, r in enumerate(results):

    ax = axes.flatten()[idx]

    # Predict curve
    Y_pred = r['model'].predict(
        r['poly'].transform(X_range)
    )

    # Training points
    ax.scatter(
        X_train,
        Y_train,
        c='blue',
        alpha=0.5,
        s=30,
        label='Train'
    )

    # Testing points
    ax.scatter(
        X_test,
        Y_test,
        c='red',
        alpha=0.7,
        s=50,
        marker='*',
        label='Test'
    )

    # Regression curve
    ax.plot(
        X_range,
        Y_pred,
        'g-',
        linewidth=2,
        label=f'Degree {r["deg"]}'
    )

    ax.set_title(
        f'Degree {r["deg"]} '
        f'(R² = {r["test_r2"]:.3f})'
    )

    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# =========================================================
# PERFORMANCE COMPARISON
# =========================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

degs = [r['deg'] for r in results]

# R² Comparison
ax1.plot(
    degs,
    [r['train_r2'] for r in results],
    'o-',
    label='Train R²',
    linewidth=2,
    markersize=8
)

ax1.plot(
    degs,
    [r['test_r2'] for r in results],
    's-',
    label='Test R²',
    linewidth=2,
    markersize=8
)

ax1.axvline(
    x=best['deg'],
    color='red',
    linestyle='--',
    alpha=0.5,
    label='Best'
)

ax1.set_xlabel('Polynomial Degree')
ax1.set_ylabel('R² Score')
ax1.set_title('R² vs Degree')

ax1.legend()
ax1.grid(True, alpha=0.3)

# RMSE Comparison
ax2.plot(
    degs,
    [r['rmse'] for r in results],
    'o-',
    color='orange',
    linewidth=2,
    markersize=8
)

ax2.axvline(
    x=best['deg'],
    color='red',
    linestyle='--',
    alpha=0.5,
    label='Best'
)

ax2.set_xlabel('Polynomial Degree')
ax2.set_ylabel('RMSE')
ax2.set_title('RMSE vs Degree')

ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


4.navi bayes
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# =========================================================
# LOAD DATASET
# =========================================================

data = pd.read_csv(
    r"C:\Users\Admin\Downloads\GHI_Report.csv"
)

# =========================================================
# CREATE EXPERIENCE LEVEL FROM H_SCORE
# =========================================================

def level(score):

    if score <= 5.5:
        return "Low"

    elif score <= 6.5:
        return "Medium"

    else:
        return "High"


data["Experience"] = data["H_Score"].apply(level)

# =========================================================
# PLOT EXPERIENCE DISTRIBUTION
# =========================================================

data["Experience"].value_counts().plot(kind="bar")

plt.title("Experience Level Distribution")
plt.xlabel("Experience Level")
plt.ylabel("Count")

plt.show()

# =========================================================
# FEATURES AND TARGET
# =========================================================

X = data[['Economy', 'Fam', 'Health', 'Freedom']]
y = data['Experience']

# =========================================================
# SPLIT DATASET
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

# =========================================================
# NAIVE BAYES MODEL
# =========================================================

nb = GaussianNB()

nb.fit(X_train, y_train)

nb_pred = nb.predict(X_test)

# =========================================================
# KNN MODEL
# =========================================================

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

knn_pred = knn.predict(X_test)

# =========================================================
# MODEL ACCURACY
# =========================================================

nb_acc = accuracy_score(y_test, nb_pred)
knn_acc = accuracy_score(y_test, knn_pred)

print("Naive Bayes Accuracy:", nb_acc)
print("KNN Accuracy:", knn_acc)

# =========================================================
# ACCURACY COMPARISON PLOT
# =========================================================

models = ["Naive Bayes", "KNN"]
accuracy = [nb_acc, knn_acc]

plt.bar(models, accuracy)

plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")

plt.show()

# =========================================================
# CONFUSION MATRIX - NAIVE BAYES
# =========================================================

cm_nb = confusion_matrix(y_test, nb_pred)

disp_nb = ConfusionMatrixDisplay(
    confusion_matrix=cm_nb
)

disp_nb.plot()

plt.title("Naive Bayes Confusion Matrix")

plt.show()

# =========================================================
# CONFUSION MATRIX - KNN
# =========================================================

cm_knn = confusion_matrix(y_test, knn_pred)

disp_knn = ConfusionMatrixDisplay(
    confusion_matrix=cm_knn
)

disp_knn.plot()

plt.title("KNN Confusion Matrix")

plt.show()


  5.classification
  import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

data = pd.read_csv(
    r"C:\Users\Admin\Downloads\GHI_Report.csv"
)

def level(score):

    if score <= 5.5:
        return "Low"

    elif score <= 6.5:
        return "Medium"

    else:
        return "High"

data["Experience"] = data["H_Score"].apply(level)

data["Experience"].value_counts().plot(kind="bar")

plt.title("Experience Level Distribution")
plt.xlabel("Experience Level")
plt.ylabel("Count")

plt.show()

X = data[['Economy', 'Fam', 'Health', 'Freedom']]
y = data['Experience']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

nb = GaussianNB()

nb.fit(X_train, y_train)

nb_pred = nb.predict(X_test)

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

knn_pred = knn.predict(X_test)

nb_acc = accuracy_score(y_test, nb_pred)
knn_acc = accuracy_score(y_test, knn_pred)

print("Naive Bayes Accuracy:", nb_acc)
print("KNN Accuracy:", knn_acc)

models = ["Naive Bayes", "KNN"]
accuracy = [nb_acc, knn_acc]

plt.bar(models, accuracy)

plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")

plt.show()

cm_nb = confusion_matrix(y_test, nb_pred)

disp_nb = ConfusionMatrixDisplay(
    confusion_matrix=cm_nb
)

disp_nb.plot()

plt.title("Naive Bayes Confusion Matrix")

plt.show()

cm_knn = confusion_matrix(y_test, knn_pred)

disp_knn = ConfusionMatrixDisplay(
    confusion_matrix=cm_knn
)

disp_knn.plot()

plt.title("KNN Confusion Matrix")

plt.show()


  



