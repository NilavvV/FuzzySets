
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# 1. Load data and build features
df = pd.read_csv("ai4i2020.csv")

df["temp_diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]
df["speed"] = df["Rotational speed [rpm]"]
df["torque"] = df["Torque [Nm]"]
df["wear"] = df["Tool wear [min]"]
# power in watts: torque (Nm) * angular speed (rad/s)
df["power"] = df["torque"] * df["speed"] * 2 * np.pi / 60
# strain: tool wear * torque (overstrain failure depends on this product)
df["strain"] = df["wear"] * df["torque"]

features = ["temp_diff", "speed", "torque", "wear", "power", "strain"]
X = df[features]
y = df["Machine failure"]

# stratify keeps the same failure ratio in train and test (failures are rare)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print("Failure rate  train: %.3f   test: %.3f" % (y_train.mean(), y_test.mean()))


# 2. Membership functions
def low_shoulder(x, full, zero):
    """1 when x <= full, falls linearly to 0 at x = zero."""
    return np.clip((zero - x) / (zero - full), 0, 1)


def high_shoulder(x, zero, full):
    """0 when x <= zero, rises linearly to 1 at x = full."""
    return np.clip((x - zero) / (full - zero), 0, 1)


def triangle(x, a, b, c):
    """0 at a, peak 1 at b, 0 at c."""
    left = (x - a) / (b - a)
    right = (c - x) / (c - b)
    return np.clip(np.minimum(left, right), 0, 1)


# breakpoints come from TRAIN percentiles only (no peeking at test data)
def pct(col, q):
    return np.percentile(X_train[col], q)


# each variable: a dictionary of  label -> function(x)
# Low / High use shoulders, Medium uses a triangle.
# For failure-related tails (power, strain) the shoulders sit further out.
mfs = {
    "temp_diff": {
        "Low":    lambda x: low_shoulder(x, pct("temp_diff", 5), pct("temp_diff", 30)),
        "Medium": lambda x: triangle(x, pct("temp_diff", 15), pct("temp_diff", 50), pct("temp_diff", 85)),
        "High":   lambda x: high_shoulder(x, pct("temp_diff", 70), pct("temp_diff", 95)),
    },
    "speed": {
        "Low":    lambda x: low_shoulder(x, pct("speed", 5), pct("speed", 30)),
        "Medium": lambda x: triangle(x, pct("speed", 15), pct("speed", 50), pct("speed", 85)),
        "High":   lambda x: high_shoulder(x, pct("speed", 70), pct("speed", 95)),
    },
    "torque": {
        "Low":    lambda x: low_shoulder(x, pct("torque", 5), pct("torque", 30)),
        "Medium": lambda x: triangle(x, pct("torque", 15), pct("torque", 50), pct("torque", 85)),
        "High":   lambda x: high_shoulder(x, pct("torque", 70), pct("torque", 95)),
    },
    "wear": {
        "Low":    lambda x: low_shoulder(x, pct("wear", 10), pct("wear", 40)),
        "Medium": lambda x: triangle(x, pct("wear", 20), pct("wear", 50), pct("wear", 80)),
        "High":   lambda x: high_shoulder(x, pct("wear", 70), pct("wear", 90)),
    },
    "power": {
        "Low":    lambda x: low_shoulder(x, pct("power", 1), pct("power", 8)),
        "Medium": lambda x: triangle(x, pct("power", 5), pct("power", 50), pct("power", 95)),
        "High":   lambda x: high_shoulder(x, pct("power", 92), pct("power", 99)),
    },
    "strain": {
        "Low":    lambda x: low_shoulder(x, pct("strain", 10), pct("strain", 40)),
        "Medium": lambda x: triangle(x, pct("strain", 20), pct("strain", 50), pct("strain", 80)),
        "High":   lambda x: high_shoulder(x, pct("strain", 85), pct("strain", 98)),
    },
}


def fuzzify(data):
    """Turn raw values into membership degrees. Returns dict {(var, label): array}."""
    out = {}
    for var, labels in mfs.items():
        for label, fn in labels.items():
            out[(var, label)] = fn(data[var].values)
    return out


# 3. Rules -> firing strengths -> risk score
# Each rule is a list of (variable, label) conditions joined by AND.
# AND = minimum of the membership degrees.
rules = [
    ("R1 Heat failure",    [("temp_diff", "Low"), ("speed", "Low")]),
    ("R2 Power too low",   [("power", "Low")]),
    ("R3 Power too high",  [("power", "High")]),
    ("R4 Overstrain",      [("strain", "High")]),
    ("R5 Worn + loaded",   [("wear", "High"), ("torque", "High")]),
]


def rule_strengths(data):
    """Returns a (n_samples, n_rules) array of firing strengths."""
    memberships = fuzzify(data)
    strengths = []
    for name, conditions in rules:
        degrees = [memberships[c] for c in conditions]
        strengths.append(np.min(degrees, axis=0))   # AND = min
    return np.array(strengths).T


def risk_score(data):
    # OR across rules = maximum: any strongly-firing rule means high risk
    return rule_strengths(data).max(axis=1)



# 4. Pick threshold on TRAIN, evaluate on TEST

train_risk = risk_score(X_train)
test_risk = risk_score(X_test)

best_t, best_f1 = 0.5, 0
for t in np.arange(0.05, 0.96, 0.05):
    f1 = f1_score(y_train, train_risk >= t)
    if f1 > best_f1:
        best_t, best_f1 = t, f1
print("\nBest threshold on train: %.2f  (train F1 = %.3f)" % (best_t, best_f1))

fuzzy_pred = (test_risk >= best_t).astype(int)

print("\n=== FUZZY CLASSIFIER (test) ===")
print(confusion_matrix(y_test, fuzzy_pred))
print(classification_report(y_test, fuzzy_pred, digits=3))

# 5. Baseline: Decision Tree
tree = DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42)
tree.fit(X_train, y_train)
tree_pred = tree.predict(X_test)

print("=== DECISION TREE (test) ===")
print(confusion_matrix(y_test, tree_pred))
print(classification_report(y_test, tree_pred, digits=3))

# 6a. Plot membership functions
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for ax, var in zip(axes.ravel(), mfs):
    xs = np.linspace(X_train[var].min(), X_train[var].max(), 400)
    for label, fn in mfs[var].items():
        ax.plot(xs, fn(xs), label=label)
    ax.set_title(var)
    ax.set_ylabel("membership")
    ax.legend()
plt.tight_layout()
plt.savefig("membership_functions.png", dpi=150)
print("Saved membership_functions.png")


failed = X_test[y_test == 1]
idx = failed.index[np.argmax(risk_score(failed))]
sample = X_test.loc[[idx]]
s = rule_strengths(sample)[0]

print("\n=== EXPLANATION for machine index", idx, "===")
print(sample.round(2).to_string())
for (name, conditions), strength in zip(rules, s):
    print("%-20s fired at %.2f   (%s)" % (name, strength, " AND ".join("%s is %s" % c for c in conditions)))
print("Risk score = max of strengths = %.2f  -> %s" % (s.max(), "FAILURE" if s.max() >= best_t else "OK"))