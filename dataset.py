import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Number of samples
n = 1650000

# Generate synthetic features
np.random.seed(42)

df = pd.DataFrame({
    "Age": np.random.randint(18, 30, n),
    "Gender": np.random.choice(["Male", "Female"], n),
    "Education": np.random.choice(["High School", "Diploma", "Bachelors", "Masters"], n),
    "Tech_Skills": np.random.randint(1, 11, n),
    "Soft_Skills": np.random.randint(1, 11, n),
    "Experience_Years": np.random.randint(0, 6, n),
    "Job_Sector": np.random.choice(
        ["IT", "Healthcare", "Retail", "Engineering", "Customer Service"], 
        size=n, 
        p=[0.25, 0.20, 0.15, 0.25, 0.15]  # different probabilities
    )
})

# Save as CSV
df.to_csv("job_suitability.csv", index=False)

print("Dataset created! Sample:")
print(df.head())

