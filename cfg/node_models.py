import pandas as pd
import networkx as nx
from grakel import Graph, GraphKernel
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
import json
import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from transformers import (RobertaModel, BertModel, RobertaForSequenceClassification, RobertaTokenizer)
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AdamW
from sklearn.utils.class_weight import compute_class_weight
import re
import node_classifier
import graph_kernel as gk_kernel
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random_state = 42


def randon_forest(train_df, val_df, test_df, le):
    train_cfg, train_node_labels, train_graph_labels, train_vul_nodes_count, train_non_vul_nodes_count, train_total_nodes_count = node_classifier.create_cfg_for_node(
        train_df)
    print("Number node labels in training data:", len(train_node_labels))
    print("Number cfg in training data:", len(train_cfg))
    print("Number of vulnerable nodes in training data:", train_vul_nodes_count)
    print("Number of non-vulnerable nodes in training data:", train_non_vul_nodes_count)
    print("Number of total nodes in training data:", train_total_nodes_count)
    val_cfg, val_node_labels, val_graph_labels, val_vul_nodes_count, val_non_vul_nodes_count, val_total_nodes_count = node_classifier.create_cfg_for_node(
        val_df)
    print("Number node labels in training data:", len(val_node_labels))
    print("Number cfg in training data:", len(val_cfg))
    print("Number of vulnerable nodes in training data:", val_vul_nodes_count)
    print("Number of non-vulnerable nodes in training data:", val_non_vul_nodes_count)
    print("Number of total nodes in training data:", val_total_nodes_count)
    test_cfg, test_node_labels, test_graph_labels, test_vul_nodes_count, test_non_vul_nodes_count, test_total_nodes_count = node_classifier.create_cfg_for_node(
        test_df)
    print("Number node labels in testing data:", len(test_node_labels))
    print("Number cfg in testing data:", len(test_cfg))
    print("Number of vulnerable nodes in testing data:", test_vul_nodes_count)
    print("Number of non-vulnerable nodes in testing data:", test_non_vul_nodes_count)
    print("Number of total nodes in testing data:", test_total_nodes_count)

    # print(f"Hello")
    # all_labels = np.concatenate((train_node_labels, val_node_labels, test_node_labels))
    # unique_labels = np.unique(all_labels)
    # print(f"Number of unique labels in dataset: {len(unique_labels)}")

    # class_weights_dict = {0: 1.0, 1: 0.05}  # Đây là ví dụ, thay bằng giá trị của bạn
    #
    # # Khởi tạo mô hình RandomForest với class weights
    # clf_node = RandomForestClassifier(random_state=42, class_weight=class_weights_dict)
    # clf_node = RandomForestClassifier(random_state=random_state, class_weight=class_weights_dict)
    clf_node = RandomForestClassifier(random_state=random_state, class_weight='balanced')

    asembly_length = node_classifier.get_max_assembly_length(train_cfg, val_cfg, test_cfg)
    train_node_features, train_node_labels = node_classifier.extract_node_features_and_labels(train_cfg, le)
    val_node_features, val_node_labels = node_classifier.extract_node_features_and_labels(val_cfg, le)
    test_node_features, test_node_labels = node_classifier.extract_node_features_and_labels(test_cfg, le)

    print("Unique node_labels:", set(train_node_labels))
    if len(set(train_node_labels)) < 2:
        raise ValueError("Training data contains only one class. Cannot train node classifier.")

    unique, counts = np.unique(train_node_labels, return_counts=True)
    is_balanced = np.all(counts == counts[0])
    print(unique, counts, is_balanced)



    if not is_balanced:
        print("Resampled node labels: Node labels are not balanced. SMOTE is applied.")
        smote = SMOTE()
        train_node_features, train_node_labels = smote.fit_resample(train_node_features, train_node_labels)

    print("Start cross-validation for node classifier.")
    skf = StratifiedKFold(n_splits=5)

    # Tính toán F1 score cho cross-validation
    cross_val_f1_scores = cross_val_score(clf_node, train_node_features, train_node_labels, cv=skf,
                                          scoring='f1_weighted')

    # Dự đoán cross-validation
    cross_val_predictions = cross_val_predict(clf_node, train_node_features, train_node_labels, cv=skf)

    print("Cross-Validation F1 Scores:", cross_val_f1_scores)
    print("Mean Cross-Validation F1 Score:", np.mean(cross_val_f1_scores))
    print("Cross-Validation Classification Report:")
    print(classification_report(train_node_labels, cross_val_predictions, digits=5))

    # feature_names = [f"feature {i}" for i in range(train_node_features.shape[1])]

    print("Start training node classifier on entire training set.")
    clf_node.fit(train_node_features, train_node_labels)

    test_node_features_flat = np.array(test_node_features, dtype=float)

    node_pred_test = clf_node.predict(test_node_features_flat)

    print("Node Test Accuracy:", accuracy_score(test_node_labels, node_pred_test))
    # Tính F1 score
    f1 = f1_score(test_node_labels, node_pred_test, average='weighted')  # Hoặc 'macro', 'micro', tùy theo yêu cầu
    precision = precision_score(test_node_labels, node_pred_test, average='weighted')
    recall = recall_score(test_node_labels, node_pred_test, average='weighted')
    print(f"Precision: {precision:.5f}")
    print(f"Recall: {recall:.5f}")
    print(f"F1 Score: {f1:.5f}")
    print("Node Test Classification Report:")
    print(classification_report(test_node_labels, node_pred_test, digits=5))

    # importances = clf_node.feature_importances_
    # std = np.std([tree.feature_importances_ for tree in clf_node.estimators_], axis=0)
    # forest_importances = pd.Series(importances, index=feature_names)

    # fig, ax = plt.subplots()
    # forest_importances.plot.bar(yerr=std, ax=ax)
    # ax.set_title("Feature importances using MDI")
    # ax.set_ylabel("Mean decrease in impurity")
    # fig.tight_layout()

    os.makedirs('./saved_kernel', exist_ok=True)

    with open('./saved_kernel/svm_model_node.pkl', 'wb') as f:
        pickle.dump(clf_node, f)

    # Graph classification
    correct_test_indices = gk_kernel.train_graph_classifier(train_df, val_df, test_df)
    return asembly_length, clf_node, correct_test_indices

class CNN(nn.Module):
    def __init__(self, input_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        # Calculate the size of the output after the convolutions and pooling
        conv_output_size = self.calculate_conv_output_size(input_dim)

        self.fc1 = nn.Linear(conv_output_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5)

    def calculate_conv_output_size(self, input_dim):
        # Assuming the input is (batch_size, channels, input_dim)
        size = input_dim
        size = (size + 2 * 1 - 3) // 1 + 1  # conv1
        size = size // 2  # pool1
        size = (size + 2 * 1 - 3) // 1 + 1  # conv2
        size = size // 2  # pool2
        return size * 64  # 64 is the number of output channels from conv2

    def forward(self, x):
        x = x.unsqueeze(1)  # Thêm chiều kênh
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten layer
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.sigmoid(x)
def train_model_cnn(model, criterion, optimizer, train_loader, val_loader, epochs=10):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.float()
            labels = labels.float()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = inputs.float()
                labels = labels.float()

                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))
                val_loss += loss.item()

        print(f'Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss / len(val_loader)}')
def evaluate_model_cnn(model, test_loader):
    model.to(device)
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.float()

            outputs = model(inputs)
            predictions.extend(outputs.squeeze().tolist())
            true_labels.extend(labels.tolist())

    return np.array(predictions), np.array(true_labels)
def evaluate_model_test_cnn(model, test_loader):
    model.to(device)
    model.eval()
    predictions = []
    true_labels = []
    count = 0
    try:
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = inputs.float()

                outputs = model(inputs)
                predictions.extend(outputs.squeeze().tolist())
                true_labels.extend(labels.tolist())

                return np.array(predictions), np.array(true_labels)
    except:
        count += 1
        # print("error count: ", count)
        return np.array(true_labels), np.array(true_labels)
def CNN(train_df, val_df, test_df, le):

    train_cfg, train_node_labels, train_graph_labels, train_vul_nodes_count, train_non_vul_nodes_count, train_total_nodes_count = node_classifier.create_cfg_for_node(
        train_df)
    print("Number node labels in training data:", len(train_node_labels))
    print("Number cfg in training data:", len(train_cfg))
    print("Number of vulnerable nodes in training data:", train_vul_nodes_count)
    print("Number of non-vulnerable nodes in training data:", train_non_vul_nodes_count)
    print("Number of total nodes in training data:", train_total_nodes_count)
    val_cfg, val_node_labels, val_graph_labels, val_vul_nodes_count, val_non_vul_nodes_count, val_total_nodes_count = node_classifier.create_cfg_for_node(
        val_df)
    test_cfg, test_node_labels, test_graph_labels, test_vul_nodes_count, test_non_vul_nodes_count, test_total_nodes_count = node_classifier.create_cfg_for_node(
        test_df)
    print("Number node labels in testing data:", len(test_node_labels))
    print("Number cfg in testing data:", len(test_cfg))
    print("Number of vulnerable nodes in testing data:", test_vul_nodes_count)
    print("Number of non-vulnerable nodes in testing data:", test_non_vul_nodes_count)
    print("Number of total nodes in testing data:", test_total_nodes_count)

    # clf_node = RandomForestClassifier(random_state=random_state, class_weight='balanced')

    asembly_length = node_classifier.get_max_assembly_length(train_cfg, val_cfg, test_cfg)
    train_node_features, train_node_labels = node_classifier.extract_node_features_and_labels(train_cfg, le)
    val_node_features, val_node_labels = node_classifier.extract_node_features_and_labels(val_cfg, le)
    test_node_features, test_node_labels = node_classifier.extract_node_features_and_labels(test_cfg, le)

    print("Unique node_labels:", set(train_node_labels))
    if len(set(train_node_labels)) < 2:
        raise ValueError("Training data contains only one class. Cannot train node classifier.")

    unique, counts = np.unique(train_node_labels, return_counts=True)
    is_balanced = np.all(counts == counts[0])
    print(unique, counts, is_balanced)

    if not is_balanced:
        print("Resampled node labels: Node labels are not balanced. SMOTE is applied.")
        smote = SMOTE()
        train_node_features, train_node_labels = smote.fit_resample(train_node_features, train_node_labels)

    train_node_features = np.array(train_node_features, dtype=float)
    train_node_labels = np.array(train_node_labels, dtype=int)
    val_node_features = np.array(val_node_features, dtype=float)
    val_node_labels = np.array(val_node_labels, dtype=int)
    test_node_features = np.array(test_node_features, dtype=float)
    test_node_labels = np.array(test_node_labels, dtype=int)

    train_dataset = TensorDataset(torch.tensor(train_node_features).float(), torch.tensor(train_node_labels).float())
    val_dataset = TensorDataset(torch.tensor(val_node_features).float(), torch.tensor(val_node_labels).float())
    test_dataset = TensorDataset(torch.tensor(test_node_features).float(), torch.tensor(test_node_labels).float())

    train_loader = DataLoader(train_dataset, batch_size=4097152, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4097152, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4097152, shuffle=False)

    input_dim = train_node_features.shape[1]

    # Cross-Validation
    # print("Start cross-validation.")
    # kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # cv_scores = []
    #
    # for train_index, val_index in kf.split(train_node_features):
    #     x_train_fold, x_val_fold = train_node_features[train_index], train_node_features[val_index]
    #     y_train_fold, y_val_fold = train_node_labels[train_index], train_node_labels[val_index]
    #
    #     train_fold_dataset = TensorDataset(torch.tensor(x_train_fold).float(), torch.tensor(y_train_fold).float())
    #     val_fold_dataset = TensorDataset(torch.tensor(x_val_fold).float(), torch.tensor(y_val_fold).float())
    #     train_fold_loader = DataLoader(train_fold_dataset, batch_size=32, shuffle=True)
    #     val_fold_loader = DataLoader(val_fold_dataset, batch_size=32, shuffle=False)
    #
    #     model = CNN(input_dim).to(device)
    #     criterion = nn.BCELoss()
    #     optimizer = optim.Adam(model.parameters(), lr=0.001)
    #
    #     train_model(model, criterion, optimizer, train_fold_loader, val_fold_loader, epochs=10)
    #
    #     val_predictions, val_true_labels = evaluate_model(model, val_fold_loader)
    #     val_predictions = (val_predictions > 0.5).astype(int)
    #     accuracy = accuracy_score(val_true_labels, val_predictions)
    #     cv_scores.append(accuracy)
    #
    # print("Cross-Validation Accuracy Scores:", cv_scores)
    # print("Mean Cross-Validation Accuracy:", np.mean(cv_scores))

    # Train the final model on the full training data
    print("Start training node classifier on full training data.")
    model = CNN(input_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    train_model_cnn(model, criterion, optimizer, train_loader, val_loader, epochs=100)

    test_predictions, test_true_labels = evaluate_model_cnn(model, test_loader)
    test_predictions_binary = (test_predictions > 0.5).astype(int)
    f1 = f1_score(test_true_labels, test_predictions_binary, average='weighted')

    print("Node Test F1 Score:", f1)
    print("Node Test Classification Report:")
    print(classification_report(test_true_labels, test_predictions_binary, digits=5))
    # print("Node Test Accuracy:", accuracy_score(test_true_labels, test_predictions))
    os.makedirs('./saved_kernel', exist_ok=True)
    os.makedirs('./saved_kernel', exist_ok=True)

    torch.save(model.state_dict(), './saved_kernel/torch_model_node.pth')
    # Graph classification
    correct_test_indices = gk_kernel.train_graph_classifier(train_df, val_df, test_df)

    return asembly_length, model, correct_test_indices

def train_xgboost(train_features, train_labels, val_features, val_labels):
    model = xgb.XGBClassifier(objective="binary:logistic", random_state=random_state)
    model.fit(train_features, train_labels, eval_set=[(val_features, val_labels)], early_stopping_rounds=10, verbose=True)
    return model

def evaluate_model_xgboost(model, test_features, test_labels):
    predictions = model.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    report = classification_report(test_labels, predictions, digits=5)
    # Tính F1 Score
    f1 = f1_score(test_labels, predictions, average='weighted')
    return accuracy, report, f1

def xgboost(train_df, val_df, test_df, le):
    train_cfg, train_node_labels, train_graph_labels, train_vul_nodes_count, train_non_vul_nodes_count, train_total_nodes_count = node_classifier.create_cfg_for_node(train_df)
    print("Number node labels in training data:", len(train_node_labels))
    print("Number cfg in training data:", len(train_cfg))
    print("Number of vulnerable nodes in training data:", train_vul_nodes_count)
    print("Number of non-vulnerable nodes in training data:", train_non_vul_nodes_count)
    print("Number of total nodes in training data:", train_total_nodes_count)
    val_cfg, val_node_labels, val_graph_labels, val_vul_nodes_count, val_non_vul_nodes_count, val_total_nodes_count = node_classifier.create_cfg_for_node(val_df)
    test_cfg, test_node_labels, test_graph_labels, test_vul_nodes_count, test_non_vul_nodes_count, test_total_nodes_count = node_classifier.create_cfg_for_node(test_df)
    print("Number node labels in testing data:", len(test_node_labels))
    print("Number cfg in testing data:", len(test_cfg))
    print("Number of vulnerable nodes in testing data:", test_vul_nodes_count)
    print("Number of non-vulnerable nodes in testing data:", test_non_vul_nodes_count)
    print("Number of total nodes in testing data:", test_total_nodes_count)

    asembly_length = node_classifier.get_max_assembly_length(train_cfg, val_cfg, test_cfg)
    train_node_features, train_node_labels = node_classifier.extract_node_features_and_labels(train_cfg, le)
    val_node_features, val_node_labels = node_classifier.extract_node_features_and_labels(val_cfg, le)
    test_node_features, test_node_labels = node_classifier.extract_node_features_and_labels(test_cfg, le)

    print("Unique node_labels:", set(train_node_labels))
    if len(set(train_node_labels)) < 2:
        raise ValueError("Training data contains only one class. Cannot train node classifier.")

    unique, counts = np.unique(train_node_labels, return_counts=True)
    is_balanced = np.all(counts == counts[0])
    print(unique, counts, is_balanced)

    if not is_balanced:
        print("Resampled node labels: Node labels are not balanced. SMOTE is applied.")
        smote = SMOTE()
        train_node_features, train_node_labels = smote.fit_resample(train_node_features, train_node_labels)

    print("Start training node classifier.")
    xgb_model = train_xgboost(train_node_features, train_node_labels, val_node_features, val_node_labels)

    test_node_features_flat = np.array(test_node_features, dtype=float)
    test_node_labels_flat = np.array(test_node_labels, dtype=int)

    test_accuracy, test_report, test_f1 = evaluate_model_xgboost(xgb_model, test_node_features_flat, test_node_labels_flat)
    print("Node Test Accuracy:", test_accuracy)
    print("Node Test F1 Score:", test_f1)
    print("Node Test Classification Report:")
    print(test_report)

    os.makedirs('./saved_kernel', exist_ok=True)

    with open('./saved_kernel/xgb_model_node.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)

    # Graph classification
    correct_test_indices = gk_kernel.train_graph_classifier(train_df, val_df, test_df)

    return asembly_length, xgb_model, correct_test_indices


class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.output(x))
        return x


def train_model_mlp(model, criterion, optimizer, train_loader, val_loader, epochs=10):
    model.to(device)  # Chuyển mô hình sang GPU
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Chuyển tensor sang GPU
            inputs = inputs.float()  # Ensure inputs are float
            labels = labels.float()  # Ensure labels are float

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)  # Chuyển tensor sang GPU
                inputs = inputs.float()  # Ensure inputs are float
                labels = labels.float()  # Ensure labels are float

                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))
                val_loss += loss.item()

        print(f'Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss / len(val_loader)}')


def evaluate_model_mlp(model, test_loader):
    model.to(device)  # Chuyển mô hình sang GPU
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Chuyển tensor sang GPU
            inputs = inputs.float()  # Ensure inputs are float

            outputs = model(inputs)
            predictions.extend(outputs.squeeze().tolist())
            true_labels.extend(labels.tolist())

    return np.array(predictions), np.array(true_labels)