import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.model import train_model
from src.utility import save_model

# Synthetic dataset
np.random.seed(42)

# Generate 50 examples
ages = np.random.randint(18, 60, 50)
genders = np.random.choice(["Male", "Female"], 50)
educations = np.random.choice(["High School", "Bachelors", "Masters"], 50)
skills = np.random.choice(["Beginner", "Intermediate", "Advanced"], 50)
experiences = np.random.randint(0, 15, 50)

X = np.column_stack((ages, genders, educations, skills, experiences))

# Target variable: simple rule for demonstration
# Not suitable: Beginner OR experience < 2 years
y = np.array([
    0 if skill == "Beginner" or exp < 2 else 1
    for skill, exp in zip(skills, experiences)
])

# Create encoders for categorical columns
encoders = {
    "Gender": LabelEncoder().fit(["Male", "Female"]),
    "Education": LabelEncoder().fit(["High School", "Bachelors", "Masters"]),
    "Skill_Level": LabelEncoder().fit(["Beginner", "Intermediate", "Advanced"])
}

# Encode categorical columns
for i, col_name in enumerate(["Gender", "Education", "Skill_Level"], start=1):
    X_col = [row[i] for row in X]
    X_encoded = encoders[col_name].transform(X_col)
    for j, row in enumerate(X):
        row[i] = X_encoded[j]

X = np.array(X, dtype=float)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestClassifier(random_state=42)
model, acc = train_model(model, X_scaled, y)
print(f"Trained model accuracy: {acc:.2f}")

# Save model
save_model(model, scaler, encoders)
