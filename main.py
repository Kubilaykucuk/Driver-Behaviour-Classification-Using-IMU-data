import numpy as np
import glob
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt

data = pd.read_csv('./crash/data.csv',index_col=0)
labels = data['class'].unique()

data_featured = pd.DataFrame()
data_prossed = pd.DataFrame()
labels = data['class'].unique()

for col in np.array([0,1,2,4,5,6,7,8,9]):    
    for label in labels:
        data_label = data[data['class']==label]
        acc_label_x = data_label.iloc[:,col]
        corr = signal.correlate(acc_label_x,np.ones(len(acc_label_x)),mode='same') / len(acc_label_x)
        clock= np.arange(64, len(acc_label_x), 128)
        data_featured=pd.concat([data_featured,pd.DataFrame(corr)], ignore_index=True)
    data_prossed = pd.concat([data_prossed,data_featured],axis=1,ignore_index=True)
    data_featured = pd.DataFrame()

data_prossed['class'] = data['class']
x = data_prossed.drop(["class"],axis=1)
y = data_prossed["class"].values

from sklearn.preprocessing import LabelEncoder
y = LabelEncoder().fit_transform(y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, shuffle=True)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

classifiers = {
    'KNN': KNeighborsClassifier(n_neighbors=3, n_jobs=-1),
    'Naive Bayes': GaussianNB(),
    'SGD': SGDClassifier(n_jobs=-1, verbose=True),
    'Random Forest': RandomForestClassifier(n_jobs=-1, random_state=1),
    'Logistic Regression': LogisticRegression(max_iter=1000, n_jobs=-1),
    'Decision Tree': DecisionTreeClassifier(random_state=1)
}

train_accuracies = {}
test_accuracies = {}

# Train and evaluate each classifier
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    train_accuracies[name] = clf.score(X_train, y_train)
    test_accuracies[name] = clf.score(X_test, y_test)
    print(f'{name} accuracy (train): {train_accuracies[name]:.4f}')
    print(f'{name} accuracy (test): {test_accuracies[name]:.4f}')

# Plot accuracies
fig, ax = plt.subplots()
labels = list(classifiers.keys())
train_scores = list(train_accuracies.values())
test_scores = list(test_accuracies.values())

x = np.arange(len(labels))
width = 0.35

ax.bar(x - width/2, train_scores, width, label='Train Accuracy')
ax.bar(x + width/2, test_scores, width, label='Test Accuracy')

ax.set_xlabel('Classifier')
ax.set_ylabel('Accuracy')
ax.set_title('Train and Test Accuracies by Classifier')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45)
ax.legend()

# Save and display plot
plt.tight_layout()
plt.savefig('classifier_accuracies.png')
plt.show()

rf = classifiers['Random Forest']
# Randomly select 4 samples and predict
sample_indices = np.random.choice(np.arange(len(y_test)), 9, replace=False)
sample_data = X_test.iloc[sample_indices]
sample_actual_labels = y_test[sample_indices]
sample_predicted_labels = rf.predict(sample_data)

# Visualize feature signals for selected samples
fig, axs = plt.subplots(3, 3, figsize=(15, 12))
axs = axs.flatten()

for i, idx in enumerate(sample_indices):
    sample_features = sample_data.iloc[i].values
    
    # Normalize features to bring out the differences
    sample_features_normalized = (sample_features - np.min(sample_features)) / (np.max(sample_features) - np.min(sample_features))
    
    # Apply PCA to reduce dimensions if needed for clearer separation (optional)
    # pca = PCA(n_components=2)
    # sample_features_normalized = pca.fit_transform(sample_features_normalized.reshape(1, -1))
    
    # Create a time-series plot for each sample
    axs[i].plot(sample_features_normalized, label='Signal Data', color='b', lw=2, alpha=0.7)
    
    # Add annotations for predicted and actual labels
    actual_label = sample_actual_labels[i]
    predicted_label = sample_predicted_labels[i]
    
    # Mark the difference between the predicted and actual labels with distinct markers
    if predicted_label != actual_label:
        axs[i].scatter(np.arange(len(sample_features_normalized)), sample_features_normalized, color='r', marker='x', label='Prediction Error')
    
    axs[i].set_title(f"Predicted: {predicted_label}, Actual: {actual_label}")
    
    # Annotating prediction and actual label on the plot
    axs[i].annotate(f'Pred: {predicted_label}', xy=(0.8, 0.9), xycoords='axes fraction', color='g', fontsize=12, weight='bold')
    axs[i].annotate(f'Act: {actual_label}', xy=(0.8, 0.8), xycoords='axes fraction', color='r', fontsize=12, weight='bold')
    
    # Add labels and grid
    axs[i].set_xlabel('Feature Index')
    axs[i].set_ylabel('Normalized Feature Value')
    axs[i].grid(True)

# Tight layout for clean visualization
plt.tight_layout()

# Save and show the plot
plt.savefig('rf_signal_predictions_with_diff.png')
plt.show()