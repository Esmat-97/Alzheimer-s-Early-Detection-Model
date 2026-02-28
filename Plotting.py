import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# عدد المرضى
n = 300
np.random.seed(42)

# توليد بيانات صناعية
data = pd.DataFrame({
    "Age": np.random.randint(55, 85, n),
    "Gender": np.random.choice(["Male", "Female"], n),
    "MMSE": np.random.normal(27, 3, n).clip(10, 30),
    "BDNF": np.random.normal(15, 5, n).clip(5, 30),
    "Tau": np.random.normal(200, 50, n).clip(100, 400),
    "AmyloidBeta": np.random.normal(600, 100, n).clip(400, 900),
    "Diabetes": np.random.choice([0,1], n, p=[0.7,0.3]),
    "Hypertension": np.random.choice([0,1], n, p=[0.6,0.4]),
})

# توليد التشخيص بناءً على MMSE + biomarkers
conditions = []
for i in range(n):
    if data.loc[i,"MMSE"] > 26 and data.loc[i,"AmyloidBeta"] < 700:
        conditions.append(0)  # Healthy
    elif 23 <= data.loc[i,"MMSE"] <= 26:
        conditions.append(1)  # MCI
    else:
        conditions.append(2)  # Alzheimer

data["Diagnosis"] = conditions


data.to_csv("synthetic_adni_data.csv", index=False)


df = pd.read_csv("synthetic_adni_data.csv")
print(df.head())






plt.figure(figsize=(12,6))

# Boxplot
plt.subplot(1,2,1)
sns.boxplot(x='Diagnosis', y='Tau', data=data)
plt.xticks([0,1,2], ['Healthy','MCI','Alzheimer'])
plt.xlabel("Diagnosis")
plt.ylabel("Tau Score")
plt.title("Tau Distribution (Boxplot)")

# Violin Plot
plt.subplot(1,2,2)
sns.violinplot(x='Diagnosis', y='Tau', data=data)
plt.xticks([0,1,2], ['Healthy','MCI','Alzheimer'])
plt.xlabel("Diagnosis")
plt.ylabel("Tau Score")
plt.title("Tau Distribution (Violin Plot)")



plt.tight_layout()
plt.savefig("Tau Distribution (Violin Plot).png")
plt.show()












plt.figure(figsize=(12,6))

# Boxplot
plt.subplot(1,2,1)
sns.boxplot(x='Diagnosis', y='AmyloidBeta', data=data)
plt.xticks([0,1,2], ['Healthy','MCI','Alzheimer'])
plt.xlabel("Diagnosis")
plt.ylabel("AmyloidBeta Score")
plt.title("AmyloidBeta Distribution (Boxplot)")

# Violin Plot
plt.subplot(1,2,2)
sns.violinplot(x='Diagnosis', y='AmyloidBeta', data=data)
plt.xticks([0,1,2], ['Healthy','MCI','Alzheimer'])
plt.xlabel("Diagnosis")
plt.ylabel("AmyloidBeta Score")
plt.title("AmyloidBeta Distribution (Violin Plot)")



plt.tight_layout()
plt.savefig("AmyloidBeta Distribution (Violin Plot).png")
plt.show()











plt.figure(figsize=(12,6))

# Boxplot
plt.subplot(1,2,1)
sns.boxplot(x='Diagnosis', y='BDNF', data=data)
plt.xticks([0,1,2], ['Healthy','MCI','Alzheimer'])
plt.xlabel("Diagnosis")
plt.ylabel("BDNF Score")
plt.title("BDNF Distribution (Boxplot)")

# Violin Plot
plt.subplot(1,2,2)
sns.violinplot(x='Diagnosis', y='BDNF', data=data)
plt.xticks([0,1,2], ['Healthy','MCI','Alzheimer'])
plt.xlabel("Diagnosis")
plt.ylabel("BDNF Score")
plt.title("BDNF Distribution (Violin Plot)")



plt.tight_layout()
plt.savefig("BDNF Distribution (Violin Plot).png")
plt.show()









cols = [
    'MMSE','Tau','BDNF','AmyloidBeta','Diagnosis'
]

# مصفوفة الترابط
corr = data[cols].corr()

# رسم Heatmap
plt.figure(figsize=(14,10))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Alzheimer Risk Factors")
plt.savefig("Correlation Heatmap of Alzheimer Biomarkers.png")
plt.show()




data['DiagnosisBinary'] = (data['Diagnosis'] == 2).astype(int)

data['AmyloidGroup'] = pd.qcut(data['AmyloidBeta'], 5)

amyloid_risk = data.groupby('AmyloidGroup')['DiagnosisBinary'].mean()

plt.figure(figsize=(8,5))
plt.plot(amyloid_risk.index.astype(str), amyloid_risk.values, marker='o')
plt.xticks(rotation=45)
plt.xlabel("Amyloid Beta Group")
plt.ylabel("Risk of Alzheimer")
plt.title("Amyloid Level vs Alzheimer Risk")
plt.tight_layout()
plt.savefig("Amyloid Level vs Alzheimer Risk.png")
plt.show()






import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_curve, auc

# -----------------------------
# 1. تجهيز البيانات
# -----------------------------
X = data[['Age','MMSE','BDNF','Tau','AmyloidBeta','Diabetes','Hypertension']]
y = data['Diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 2. Logistic Regression
# -----------------------------
log_model = LogisticRegression(max_iter=500, solver='lbfgs', multi_class='auto')
log_model.fit(X_train_scaled, y_train)
y_pred_log = log_model.predict(X_test_scaled)

print("Logistic Regression Report:")
print(classification_report(y_test, y_pred_log))

# -----------------------------
# 3. Random Forest
# -----------------------------
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("Random Forest Report:")
print(classification_report(y_test, y_pred_rf))

# Feature Importance
importances = rf_model.feature_importances_
for feat, imp in zip(X.columns, importances):
    print(f"{feat}: {imp:.3f}")

# -----------------------------
# 4. Pairplot Visualization
# -----------------------------
sns.pairplot(data[['Age','MMSE','BDNF','Tau','AmyloidBeta','Diagnosis']], hue="Diagnosis")
plt.show()

# -----------------------------
# 5. ROC Curve (Logistic Regression)
# -----------------------------
y_prob = log_model.predict_proba(X_test_scaled)

plt.figure(figsize=(8,6))
for i, cls in enumerate(log_model.classes_):
    fpr, tpr, _ = roc_curve(y_test==cls, y_prob[:,i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Class {cls} (AUC = {roc_auc:.2f})")

plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Logistic Regression)")
plt.legend()
plt.show()

