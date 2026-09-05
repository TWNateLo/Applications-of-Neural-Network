import csv
import numpy as np


# =========================================================
# 1. LOAD CSV + DATA PREPROCESSING
# =========================================================

print("Loading smartphone addiction data...")
filename = "Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv"

rows = []
with open(filename, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

# Will predict:
# addicted_label = 0 or 1

# Dropping the following columns:
# - transaction_id, user_id : only IDs
# - addiction_level         : very close to the target label, so I avoid using it

# Manual encoding for categorical columns
gender_map = {
    "Male": [1, 0, 0],
    "Female": [0, 1, 0],
    "Other": [0, 0, 1]
}

stress_map = {
    "Low": [1, 0, 0],
    "Medium": [0, 1, 0],
    "High": [0, 0, 1]
}

impact_map = {
    "No": [0],
    "Yes": [1]
}

X_list = []
y_list = []
original_rows = []

for row in rows:
    features = []

    # Numeric features
    features.append(float(row["age"]))
    features.append(float(row["daily_screen_time_hours"]))
    features.append(float(row["social_media_hours"]))
    features.append(float(row["gaming_hours"]))
    features.append(float(row["work_study_hours"]))
    features.append(float(row["sleep_hours"]))
    features.append(float(row["notifications_per_day"]))
    features.append(float(row["app_opens_per_day"]))
    features.append(float(row["weekend_screen_time"]))

    # Categorical features -> numeric
    features.extend(gender_map[row["gender"]])
    features.extend(stress_map[row["stress_level"]])
    features.extend(impact_map[row["academic_work_impact"]])

    X_list.append(features)
    y_list.append(int(row["addicted_label"]))
    original_rows.append(row)

X = np.array(X_list, dtype=float)
y = np.array(y_list, dtype=int)

print("Data loaded.")
print("X shape:", X.shape)
print("y shape:", y.shape)

# =========================================================
# 2. SHUFFLE + TRAIN/TEST SPLIT
# =========================================================

np.random.seed(42)
indices = np.random.permutation(len(X))

X = X[indices]
y = y[indices]
original_rows = [original_rows[i] for i in indices]

split_idx = int(0.8 * len(X))

X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
rows_train, rows_test = original_rows[:split_idx], original_rows[split_idx:]

# =========================================================
# 3. NORMALIZATION
# =========================================================

# Normalize numeric range using training data only
feature_mean = X_train.mean(axis=0)
feature_std = X_train.std(axis=0)

# Avoid division by zero
feature_std[feature_std == 0] = 1.0

X_train = (X_train - feature_mean) / feature_std
X_test = (X_test - feature_mean) / feature_std

# =========================================================
# 4. PERCEPTRON MODEL
# =========================================================

def sign(z):
    return 1 if z >= 0 else 0

def perceptron(X, weights, bias):
    scores = np.dot(X, weights) + bias
    predictions = []
    for score in scores:
        predictions.append(sign(score))
    return np.array(predictions, dtype=int)

# =========================================================
# 5. TRAINING PARAMETERS
# =========================================================

epochs = 40
bias = 0.0
lr = 0.01
weights = np.zeros(X_train.shape[1])

# =========================================================
# 6. TRAINING
# =========================================================

for epoch in range(epochs):
    updates_in_epoch = 0

    for x_i, y_i in zip(X_train, y_train):
        y_pred = perceptron(x_i.reshape(1, -1), weights, bias)[0]

        # Update only when wrong
        if y_i != y_pred:
            delta_w = lr * (y_i - y_pred) * x_i
            weights += delta_w

            delta_b = lr * (y_i - y_pred)
            bias += delta_b

            updates_in_epoch += 1

    # Epoch result
    train_preds_epoch = perceptron(X_train, weights, bias)
    train_acc_epoch = (train_preds_epoch == y_train).mean()
    print(f"Epoch {epoch+1}/{epochs} - Updates: {updates_in_epoch} - Train Acc: {train_acc_epoch:.4f}")

# =========================================================
# 7. PREDICTION + EVALUATION
# =========================================================

train_preds = perceptron(X_train, weights, bias)
test_preds = perceptron(X_test, weights, bias)

train_acc = (train_preds == y_train).mean()
test_acc = (test_preds == y_test).mean()

print("\n=== Final Results ===")
print("Trained Weights:", weights)
print("Trained Bias:", bias)
print(f"Train Acc: {train_acc:.4f}")
print(f"Test Acc: {test_acc:.4f}")

# =========================================================
# 8. SHOW SOME CORRECT / WRONG EXAMPLES
# =========================================================

def label_text(v):
    return "Addicted(1)" if v == 1 else "Not Addicted(0)"

correct_indices = []
wrong_indices = []

for i in range(len(X_test)):
    if test_preds[i] == y_test[i]:
        correct_indices.append(i)
    else:
        wrong_indices.append(i)

print("\nNumber of correct test predictions:", len(correct_indices))
print("Number of wrong test predictions:", len(wrong_indices))

print("\n=== Some Correct Predictions ===")
for idx in correct_indices[:5]:
    row = rows_test[idx]
    print("-" * 60)
    print("True Label :", label_text(y_test[idx]))
    print("Pred Label :", label_text(test_preds[idx]))
    print("age:", row["age"])
    print("gender:", row["gender"])
    print("daily_screen_time_hours:", row["daily_screen_time_hours"])
    print("social_media_hours:", row["social_media_hours"])
    print("gaming_hours:", row["gaming_hours"])
    print("work_study_hours:", row["work_study_hours"])
    print("sleep_hours:", row["sleep_hours"])
    print("notifications_per_day:", row["notifications_per_day"])
    print("app_opens_per_day:", row["app_opens_per_day"])
    print("weekend_screen_time:", row["weekend_screen_time"])
    print("stress_level:", row["stress_level"])
    print("academic_work_impact:", row["academic_work_impact"])

print("\n=== Some Wrong Predictions ===")
for idx in wrong_indices[:5]:
    row = rows_test[idx]
    print("-" * 60)
    print("True Label :", label_text(y_test[idx]))
    print("Pred Label :", label_text(test_preds[idx]))
    print("age:", row["age"])
    print("gender:", row["gender"])
    print("daily_screen_time_hours:", row["daily_screen_time_hours"])
    print("social_media_hours:", row["social_media_hours"])
    print("gaming_hours:", row["gaming_hours"])
    print("work_study_hours:", row["work_study_hours"])
    print("sleep_hours:", row["sleep_hours"])
    print("notifications_per_day:", row["notifications_per_day"])
    print("app_opens_per_day:", row["app_opens_per_day"])
    print("weekend_screen_time:", row["weekend_screen_time"])
    print("stress_level:", row["stress_level"])
    print("academic_work_impact:", row["academic_work_impact"])