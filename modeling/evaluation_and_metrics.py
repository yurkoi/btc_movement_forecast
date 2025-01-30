import json
import matplotlib.pyplot as plt
import numpy as np

with open("predictions_metrics.json", "r") as file:
    data = json.load(file)

labels = [entry["Prediction Label"] for entry in data]
accuracies = [entry["Accuracy"] for entry in data]
f1_scores = [entry["F1 Score"] for entry in data]
roc_aucs = [entry["ROC-AUC Score"] for entry in data]
precisions = [entry["Precision"] for entry in data]
recalls = [entry["Recall"] for entry in data]

x = np.arange(len(labels))
width = 0.15

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - 2 * width, accuracies, width, label='Accuracy')
rects2 = ax.bar(x - width, f1_scores, width, label='F1 Score')
rects3 = ax.bar(x, roc_aucs, width, label='ROC-AUC')
rects4 = ax.bar(x + width, precisions, width, label='Precision')
rects5 = ax.bar(x + 2 * width, recalls, width, label='Recall')

ax.set_xlabel('Prediction Label')
ax.set_title('Model Performance Metrics by Prediction Label')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

add_labels(rects1)
add_labels(rects2)
add_labels(rects3)
add_labels(rects4)
add_labels(rects5)

plt.tight_layout()
plt.savefig("model_performance_metrics.png", format="png", dpi=300)


import json

with open("predictions_metrics.json", "r") as file:
    data = json.load(file)

def get_params_by_label(data, label):
    for item in data:
        if item["Prediction Label"] == label:
            return item["Best Parameters"]
    return None


temp_results = {
        "1d_pred": [],
        "3d_pred": [],
        "5d_pred": [],
        "1d_pred_proba": [],
        "3d_pred_proba": [],
        "5d_pred_proba": [],
        "1d_actual": [],
        "3d_actual": [],
        "5d_actual": [],
}


for days in ["1d", "3d", "5d"]:
    label = f"is_positive_growth_{days}_future"
    best_params = get_params_by_label(data, label)
    print(f"Best Parameters for '{label}':\n{best_params}")

    X_train, y_train, X_test, y_test = final_preprocess_dataframe(new_df, features_list,
                                                                split_date,
                                                                label)

    best_model = XGBClassifier(**best_params, random_state=123,
                            use_label_encoder=True, enable_categorical=True)
    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    temp_results[f'{days}_pred'].extend(y_pred)
    temp_results[f'{days}_pred_proba'].extend(y_pred_proba)
    temp_results[f'{days}_actual'].extend(y_test)


results_df = pd.DataFrame(temp_results)
results_df.head(5)