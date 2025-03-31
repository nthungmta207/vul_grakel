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
from joblib import Parallel, delayed
import pickle
import os
import json
import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from transformers import (RobertaModel, BertModel, RobertaForSequenceClassification, RobertaTokenizer)
from sklearn.model_selection import KFold, cross_val_score
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AdamW
import re
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import graph_kernel as gk_kernel


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Set random seed for reproducibility
random_state = 42
# Kiểm tra xem GPU có sẵn không
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = RobertaModel.from_pretrained('roberta-base')
# tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')

def load_data(train_file, val_file, test_file):
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)
    return train_df, val_df, test_df


def find_max_length(*dfs):
    max_lengths = []
    for df in dfs:
        for _, row in df.iterrows():
            cfg_json = row.get('processed_func_cfg') or row.get('vul_func_with_fix_cfg')
            if pd.notna(cfg_json):
                cfg_data = json.loads(cfg_json)
                for block in cfg_data['blocks']:
                    assembly = block.get('assembly', [])
                    max_lengths.append(len(extract_opcodes(assembly).split('\n')))
    return max(max_lengths)

def create_cfg_for_node(df):
    cfg_list = []
    node_labels = []
    graph_labels = []
    vul_node_count = 0
    non_vul_node_count = 0
    total_node_count = 0
    for _, row in df.iterrows():
        try:
            if 'processed_func_cfg' in row and pd.notna(row['processed_func_cfg']):
                cfg_data = json.loads(row['processed_func_cfg'])
                flaw_line_address = row['flaw_line_address']
                cwe_id = row['CWE ID']
                graph_label = 1  # Vulnerable graph

                cfg = nx.DiGraph()
                for block in cfg_data['blocks']:
                    address = block.get('address', None)
                    if address is not None:
                        if address == flaw_line_address:
                            is_vulnerable = 1
                            node_label = f"{cwe_id}_{is_vulnerable}"
                            cfg.add_node(address, label=node_label,
                                         assembly=block.get('assembly', []))
                           
                            node_labels.append((address, node_label))
                            vul_node_count += 1
                            total_node_count += 1
                        else:
                            is_vulnerable = 0
                            node_label = f"{cwe_id}_{is_vulnerable}"
                            cfg.add_node(address, label=node_label,
                                         assembly=block.get('assembly', []))
                            total_node_count += 1
                    else:
                        logging.warning(f"Block without address found: {block}")

                for edge in cfg_data['edges']:
                    cfg.add_edge(edge['from'], edge['to'])

                for node in cfg.nodes():
                    if 'label' not in cfg.nodes[node]:
                        cfg.nodes[node]['label'] = f"{cwe_id}_0"
                    if 'assembly' not in cfg.nodes[node]:
                        cfg.nodes[node]['assembly'] = []

                cfg_list.append(cfg)
                graph_labels.append(graph_label)

            if 'vul_func_with_fix_cfg' in row and pd.notna(row['vul_func_with_fix_cfg']):
                cfg_data = json.loads(row['vul_func_with_fix_cfg'])
                line_after_address = row['line_after_address']
                cwe_id = row['CWE ID']
                graph_label = 0  # Non-vulnerable graph

                cfg = nx.DiGraph()
                for block in cfg_data['blocks']:
                    address = block.get('address', None)
                    if address is not None:
                        if address == line_after_address:
                            is_vulnerable = 0
                            node_label = f"{cwe_id}_{is_vulnerable}"
                            cfg.add_node(address, label=node_label,
                                         assembly=block.get('assembly', []))
                            node_labels.append((address, node_label))
                            non_vul_node_count += 1
                            total_node_count += 1
                        else:
                            is_vulnerable = 0
                            node_label = f"{cwe_id}_{is_vulnerable}"
                            cfg.add_node(address, label=node_label,
                                         assembly=block.get('assembly', []))
                            total_node_count += 1
                    else:
                        logging.warning(f"Block without address found: {block}")

                for edge in cfg_data['edges']:
                    cfg.add_edge(edge['from'], edge['to'])

                for node in cfg.nodes():
                    if 'label' not in cfg.nodes[node]:
                        cfg.nodes[node]['label'] = f"{cwe_id}_0"
                    if 'assembly' not in cfg.nodes[node]:
                        cfg.nodes[node]['assembly'] = []

                cfg_list.append(cfg)
                graph_labels.append(graph_label)

        except Exception as e:
            logging.error(f"Error processing row: {e}")

    return cfg_list, node_labels, graph_labels, vul_node_count, non_vul_node_count, total_node_count
def create_cfg_for_graph(df):
    cfg_list = []
    node_labels = []
    graph_labels = []
    vul_node_count = 0
    non_vul_node_count = 0
    total_node_count = 0
    for _, row in df.iterrows():
        try:
            if 'processed_func_cfg' in row and pd.notna(row['processed_func_cfg']):
                cfg_data = json.loads(row['processed_func_cfg'])
                flaw_line_address = row['flaw_line_address']
                cwe_id = row['CWE ID']
                graph_label = 1  # Vulnerable graph

                cfg = nx.DiGraph()
                for block in cfg_data['blocks']:
                    address = block.get('address', None)
                    if address is not None:
                        if address == flaw_line_address:
                            is_vulnerable = 0
                            node_label = f"{cwe_id}_{is_vulnerable}"
                            cfg.add_node(address, label=node_label,
                                         assembly=block.get('assembly', []))
                            node_labels.append((address, node_label))
                            vul_node_count += 1
                            total_node_count += 1
                        else:
                            is_vulnerable = 0
                            node_label = f"{cwe_id}_{is_vulnerable}"
                            cfg.add_node(address, label=node_label,
                                         assembly=block.get('assembly', []))
                            total_node_count += 1
                    else:
                        logging.warning(f"Block without address found: {block}")

                for edge in cfg_data['edges']:
                    cfg.add_edge(edge['from'], edge['to'])

                for node in cfg.nodes():
                    if 'label' not in cfg.nodes[node]:
                        cfg.nodes[node]['label'] = f"{cwe_id}_0"
                    if 'assembly' not in cfg.nodes[node]:
                        cfg.nodes[node]['assembly'] = []

                cfg_list.append(cfg)
                graph_labels.append(graph_label)

            if 'vul_func_with_fix_cfg' in row and pd.notna(row['vul_func_with_fix_cfg']):
                cfg_data = json.loads(row['vul_func_with_fix_cfg'])
                line_after_address = row['line_after_address']
                cwe_id = row['CWE ID']
                graph_label = 0  # Non-vulnerable graph

                cfg = nx.DiGraph()
                for block in cfg_data['blocks']:
                    address = block.get('address', None)
                    if address is not None:
                        if address == line_after_address:
                            is_vulnerable = 0
                            node_label = f"{cwe_id}_{is_vulnerable}"
                            cfg.add_node(address, label=node_label,
                                         assembly=block.get('assembly', []))
                            node_labels.append((address, node_label))
                            non_vul_node_count += 1
                            total_node_count += 1
                        else:
                            is_vulnerable = 0
                            node_label = f"{cwe_id}_{is_vulnerable}"
                            cfg.add_node(address, label=node_label,
                                         assembly=block.get('assembly', []))
                            total_node_count += 1
                    else:
                        logging.warning(f"Block without address found: {block}")

                for edge in cfg_data['edges']:
                    cfg.add_edge(edge['from'], edge['to'])

                for node in cfg.nodes():
                    if 'label' not in cfg.nodes[node]:
                        cfg.nodes[node]['label'] = f"{cwe_id}_0"
                    if 'assembly' not in cfg.nodes[node]:
                        cfg.nodes[node]['assembly'] = []

                cfg_list.append(cfg)
                graph_labels.append(graph_label)

        except Exception as e:
            logging.error(f"Error processing row: {e}")

    return cfg_list, node_labels, graph_labels, vul_node_count, non_vul_node_count, total_node_count
def create_cfg_for_top_k(df):
    cfg_list = []
    node_labels = []
    graph_labels = []

    for _, row in df.iterrows():
        try:
            if 'processed_func_cfg' in row and pd.notna(row['processed_func_cfg']):
                cfg_data = json.loads(row['processed_func_cfg'])
                flaw_line_address = row['flaw_line_address']
                cwe_id = row['CWE ID']
                graph_label = 1  # Vulnerable graph

                cfg = nx.DiGraph()
                for block in cfg_data['blocks']:
                    address = block.get('address', None)
                    if address is not None:
                        is_vulnerable = 1 if address == flaw_line_address else 0
                        node_label = f"{cwe_id}_{is_vulnerable}"
                        cfg.add_node(address, label=node_label, bytes=block.get('bytes', ''),
                                     assembly=block.get('assembly', []))
                        node_labels.append((address, node_label))
                    else:
                        logging.warning(f"Block without address found: {block}")

                for edge in cfg_data['edges']:
                    cfg.add_edge(edge['from'], edge['to'])

                for node in cfg.nodes():
                    if 'label' not in cfg.nodes[node]:
                        cfg.nodes[node]['label'] = f"{cwe_id}_0"
                    if 'assembly' not in cfg.nodes[node]:
                        cfg.nodes[node]['assembly'] = []
                # print(cfg)
                cfg_list.append(cfg)
                graph_labels.append(graph_label)

        except Exception as e:
            logging.error(f"Error processing row: {e}")

    return cfg_list, node_labels, graph_labels

def convert_cfg_to_grakel(cfg):
    adjacency_matrix = nx.adjacency_matrix(cfg).todense()
    node_labels = {i: cfg.nodes[node]['label'] for i, node in enumerate(cfg.nodes)}
    return Graph(adjacency_matrix, node_labels=node_labels)

def process_in_batches(data, labels, batch_size):
    num_batches = (len(data) + batch_size - 1) // batch_size
    print(f"Processing data in {num_batches} batches")
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size], labels[i:i + batch_size]

def get_max_assembly_length(cfg_list_1,  cfg_list_2, cfg_list_3):
    assembly_lengths = []
    for cfg in cfg_list_1:
        for node in cfg.nodes(data=True):
            attributes = node[1]
            assembly = attributes.get('assembly', [])
            assembly_lengths.append(len(assembly))
    for cfg in cfg_list_2:
        for node in cfg.nodes(data=True):
            attributes = node[1]
            assembly = attributes.get('assembly', [])
            assembly_lengths.append(len(assembly))
    for cfg in cfg_list_3:
        for node in cfg.nodes(data=True):
            attributes = node[1]
            assembly = attributes.get('assembly', [])
            assembly_lengths.append(len(assembly))
    max_assembly_length = max(assembly_lengths)
    return max_assembly_length

def get_max_assembly_length_top_k(cfg_list):
    assembly_lengths = []
    for cfg in cfg_list:
        for node in cfg.nodes(data=True):
            attributes = node[1]
            assembly = attributes.get('assembly', [])
            assembly_lengths.append(len(assembly))

    max_assembly_length = max(assembly_lengths)
    return max_assembly_length

def extract_opcode_operand(assembly_lines):
    result = ""
    for assembly_line in assembly_lines:
        match = re.match(r"\s*0x[0-9a-fA-F]+:\s*(\w+)\s*(.*)", assembly_line)
        if match:
            opcode = match.group(1)
            operand = match.group(2).strip()
            result += f"{opcode} {operand}\n"
        else:
            result += "\n"
    return result.strip()

def extract_opcodes(assembly_lines):
    opcodes_str = ""
    for assembly_line in assembly_lines:
        match = re.match(r"\s*0x[0-9a-fA-F]+:\s*(\w+)\s*", assembly_line)
        if match:
            opcode = match.group(1)
            opcodes_str += f"{opcode}\n"
        else:
            opcodes_str += "\n"
    return opcodes_str.strip()

def extract_operands(assembly_lines):
    operands_str = ""
    for assembly_line in assembly_lines:
        match = re.match(r"\s*0x[0-9a-fA-F]+:\s*\w+\s*(.*)", assembly_line)
        if match:
            operand = match.group(1).strip()
            operands_str += f"{operand}\n"
        else:
            operands_str += "\n"
    return operands_str.strip()


def extract_node_features_and_labels(cfg_list, le, max_length):
    node_features = []
    node_labels = []
    count_vul = 0
    count_non_vul = 0

    for cfg in cfg_list:
        for node in cfg.nodes(data=True):
            address, attributes = node
            cwe_id, is_vulnerable = attributes['label'].split('_')

            if is_vulnerable == '0' or is_vulnerable == '1':
                # cwe_id_encoded = le.transform([cwe_id])[0]
                assembly = attributes.get('assembly', [])
                if len(assembly) > 0:
                    assembly_code = extract_opcodes(assembly)
                    assembly_list = assembly_code.split('\n')

                    # Mã hóa opcode thành số nguyên
                    encoded_opcodes = le.transform(assembly_list)

                    # Padding opcode để đảm bảo tất cả các đặc trưng có cùng kích thước
                    if len(encoded_opcodes) > max_length:
                        encoded_opcodes = encoded_opcodes[:max_length]  # Cắt bớt nếu quá dài
                    else:
                        padding = [0] * (max_length - len(encoded_opcodes))  # Đệm thêm 0 nếu quá ngắn
                        encoded_opcodes = list(encoded_opcodes) + padding

                    # Thêm các opcode mã hóa vào danh sách đặc trưng
                    node_features.append(encoded_opcodes)
                    node_labels.append(int(is_vulnerable))

                    if is_vulnerable == '1':
                        count_vul += 1
                    else:
                        count_non_vul += 1

    print("Number of vulnerable nodes after extraction:", count_vul)
    print("Number of non-vulnerable nodes after extraction:", count_non_vul)
    return node_features, node_labels
def extract_node_features_and_labels_for_each_function(cfg_list, le, max_length):
    all_node_features = []
    all_node_labels = []

    for cfg in cfg_list:
        node_features = []
        node_labels = []
        for node in cfg.nodes(data=True):
            address, attributes = node
            cwe_id, is_vulnerable = attributes['label'].split('_')

            if is_vulnerable == '0' or is_vulnerable == '1':
                assembly = attributes.get('assembly', [])
                if len(assembly) > 0:
                    assembly_code = extract_opcodes(assembly)
                    assembly_list = assembly_code.split('\n')

                    # Mã hóa opcode thành số nguyên
                    encoded_opcodes = le.transform(assembly_list)

                    # Padding opcode để đảm bảo tất cả các đặc trưng có cùng kích thước
                    if len(encoded_opcodes) > max_length:
                        encoded_opcodes = encoded_opcodes[:max_length]  # Cắt bớt nếu quá dài
                    else:
                        padding = [0] * (max_length - len(encoded_opcodes))  # Đệm thêm 0 nếu quá ngắn
                        encoded_opcodes = list(encoded_opcodes) + padding

                    # Thêm các opcode mã hóa vào danh sách đặc trưng
                    node_features.append(encoded_opcodes)
                    node_labels.append(int(is_vulnerable))

        all_node_features.append(node_features)
        all_node_labels.append(node_labels)

    return all_node_features, all_node_labels



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
def train_model(model, criterion, optimizer, train_loader, val_loader, epochs=10):
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
def evaluate_model(model, test_loader):
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
def evaluate_model_test(model, test_loader):
    model.to(device)
    model.eval()
    predictions = []
    true_labels = []


    # with torch.no_grad():
    #     for inputs, labels in test_loader:
    #         inputs, labels = inputs.to(device), labels.to(device)
    #         inputs = inputs.float()
    #         print(len(inputs))
    #         print(len(labels))
    #         outputs = model(inputs)
    #         print(len(outputs))
    #         print(type(outputs))
    #         if isinstance(outputs, torch.Tensor):
    #             print(outputs.squeeze().tolist())
    #             predictions.extend(outputs.squeeze().tolist())
    #         else:
    #             predictions.append(outputs)
    #
    #         true_labels.extend(labels.tolist())
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.float()

            outputs = model(inputs)
            if isinstance(outputs, torch.Tensor):
                outputs = outputs.squeeze().tolist()
                if isinstance(outputs, float):  # Ensure outputs is a list, even if it contains only one float
                    outputs = [outputs]
            else:
                outputs = [outputs]

            predictions.extend(outputs)
            true_labels.extend(labels.tolist())

    return np.array(predictions), np.array(true_labels)

def train_node_classifier(train_df, val_df, test_df, le):

    train_cfg, train_node_labels, train_graph_labels, train_vul_nodes_count, train_non_vul_nodes_count, train_total_nodes_count = create_cfg_for_node(
        train_df)
    print("Number node labels in training data:", len(train_node_labels))
    print("Number cfg in training data:", len(train_cfg))
    print("Number of vulnerable nodes in training data:", train_vul_nodes_count)
    print("Number of non-vulnerable nodes in training data:", train_non_vul_nodes_count)
    print("Number of total nodes in training data:", train_total_nodes_count)
    val_cfg, val_node_labels, val_graph_labels, val_vul_nodes_count, val_non_vul_nodes_count, val_total_nodes_count = create_cfg_for_node(
        val_df)
    test_cfg, test_node_labels, test_graph_labels, test_vul_nodes_count, test_non_vul_nodes_count, test_total_nodes_count = create_cfg_for_node(
        test_df)
    print("Number node labels in testing data:", len(test_node_labels))
    print("Number cfg in testing data:", len(test_cfg))
    print("Number of vulnerable nodes in testing data:", test_vul_nodes_count)
    print("Number of non-vulnerable nodes in testing data:", test_non_vul_nodes_count)
    print("Number of total nodes in testing data:", test_total_nodes_count)

    # clf_node = RandomForestClassifier(random_state=random_state, class_weight='balanced')

    asembly_length = get_max_assembly_length(train_cfg, val_cfg, test_cfg)
    max_length = find_max_length(train_df, val_df, test_df)

    # Tạo đặc trưng từ opcode và nhãn cho các node
    train_node_features, train_node_labels = extract_node_features_and_labels(train_cfg, le, max_length)
    val_node_features, val_node_labels = extract_node_features_and_labels(val_cfg, le, max_length)
    test_node_features, test_node_labels = extract_node_features_and_labels(test_cfg, le, max_length)

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

    # Chuyển đổi đặc trưng và nhãn sang numpy arrays
    train_node_features = np.array(train_node_features, dtype=float)
    train_node_labels = np.array(train_node_labels, dtype=int)
    val_node_features = np.array(val_node_features, dtype=float)
    val_node_labels = np.array(val_node_labels, dtype=int)
    test_node_features = np.array(test_node_features, dtype=float)
    test_node_labels = np.array(test_node_labels, dtype=int)

    # Tạo TensorDataset và DataLoader cho training, validation và testing
    train_dataset = TensorDataset(torch.tensor(train_node_features).float(), torch.tensor(train_node_labels).float())
    val_dataset = TensorDataset(torch.tensor(val_node_features).float(), torch.tensor(val_node_labels).float())
    test_dataset = TensorDataset(torch.tensor(test_node_features).float(), torch.tensor(test_node_labels).float())

    train_loader = DataLoader(train_dataset, batch_size=10000, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=10000, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=10000, shuffle=False)

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

    train_model(model, criterion, optimizer, train_loader, val_loader, epochs=100)

    test_predictions, test_true_labels = evaluate_model(model, test_loader)
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

    return asembly_length, model, correct_test_indices, max_length


def train_all():
    train_file = '../data/big-vul_dataset/train_graph.csv'
    val_file = '../data/big-vul_dataset/val_graph.csv'
    test_file = '../data/big-vul_dataset/test_graph.csv'

    # Load data
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)

    # Fit LabelEncoder với toàn bộ opcode từ tất cả các tập dữ liệu
    all_opcodes = []
    for df in [train_df, val_df, test_df]:
        for _, row in df.iterrows():
            if 'processed_func_cfg' in row and pd.notna(row['processed_func_cfg']):
                cfg_data = json.loads(row['processed_func_cfg'])
                for block in cfg_data['blocks']:
                    assembly = block.get('assembly', [])
                    assembly_code = extract_opcodes(assembly)
                    assembly_list = assembly_code.split('\n')
                    all_opcodes.extend(assembly_list)
            if 'vul_func_with_fix_cfg' in row and pd.notna(row['vul_func_with_fix_cfg']):
                cfg_data = json.loads(row['vul_func_with_fix_cfg'])
                for block in cfg_data['blocks']:
                    assembly = block.get('assembly', [])
                    assembly_code = extract_opcodes(assembly)
                    assembly_list = assembly_code.split('\n')
                    all_opcodes.extend(assembly_list)

    # Sử dụng LabelEncoder để mã hóa toàn bộ opcode
    le = LabelEncoder()
    le.fit(all_opcodes)

    # le.fit(train_df['CWE ID'].tolist() + val_df['CWE ID'].tolist() + test_df['CWE ID'].tolist())

    # train_df['CWE ID Encoded'] = le.transform(train_df['CWE ID'])
    # val_df['CWE ID Encoded'] = le.transform(val_df['CWE ID'])
    # test_df['CWE ID Encoded'] = le.transform(test_df['CWE ID'])
    # train_graph_classifier(train_df, val_df, test_df)
    assembly_length, clf_node, correct_test_indices, max_length = train_node_classifier(train_df, val_df, test_df, le)

    return assembly_length, clf_node, correct_test_indices, max_length, le

def is_label_in_top_10(labels, accuracies):
    # Combine labels and accuracies into a list of tuples
    combined = list(zip(labels, accuracies))

    # Sort the combined list by accuracy in descending order
    sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)

    # Extract the top 10 elements
    top_10 = sorted_combined[:1]

    # Check if label 1 is in the top 10
    for label, _ in top_10:
        if label == 1:
            return True
    return False
def has_matching_index(predicted_labels, true_labels):
    """
    Checks if there is at least one index where both predicted and true labels are 1.

    :param predicted_labels: List or numpy array of predicted labels.
    :param true_labels: List or numpy array of true labels.
    :return: True if such an index exists, False otherwise.
    """
    # print(predicted_labels)
    # print(true_labels)
    if len(predicted_labels) != len(true_labels):
        raise ValueError("The length of predicted_labels and true_labels must be the same.")

    for pred, true in zip(predicted_labels, true_labels):
        if pred == 1 and true == 1:
            return True

    return False
def convert_to_binary_labels(probabilities):
    """
    Converts a list of probabilities to binary labels.
    The highest probability is set to 1, and all others are set to 0.

    :param probabilities: List or numpy array of probabilities.
    :return: List of binary labels.
    """
    # Convert to numpy array if not already
    probabilities = np.array(probabilities)

    # Create a binary array with all zeros
    binary_labels = np.zeros_like(probabilities, dtype=int)

    # Find the index of the highest probability
    highest_index = np.argmax(probabilities)

    # Set the highest probability's index to 1
    binary_labels[highest_index] = 1

    return binary_labels
def test_node_top_10(asembly_length, clf_node,correct_test_indices, max_length, le):
    train_file = '../data/big-vul_dataset/train_graph.csv'
    val_file = '../data/big-vul_dataset/val_graph.csv'
    test_file = '../data/big-vul_dataset/test_graph.csv'

    train_df, val_df, test_df = load_data(train_file, val_file, test_file)

    # le.fit(train_df['CWE ID'].tolist() + val_df['CWE ID'].tolist() + test_df['CWE ID'].tolist())
    #
    # train_df['CWE ID Encoded'] = le.transform(train_df['CWE ID'])
    # val_df['CWE ID Encoded'] = le.transform(val_df['CWE ID'])
    # test_df['CWE ID Encoded'] = le.transform(test_df['CWE ID'])

    test_cfg, test_node_labels, test_graph_labels = create_cfg_for_top_k(test_df)



    print("Number node labels in all bad fucntion in testing data:", len(test_node_labels))
    print("Number cfg (bad function only) in testing data:", len(test_cfg))

    # print("Unique node labels in testing data:", set([label for _, label in test_node_labels]))
    # print("Count of each node label in testing data:",
    #       {label: sum(1 for _, l in test_node_labels if l == label) for label in
    #        set([label for _, label in test_node_labels])})

    # with open('./saved_kernel/svm_model_node.pkl', 'rb') as f:
    #     clf_node = pickle.load(f)

    print("Accuracy for all bad function in testing data:")
    test_node_features, test_node_labels = extract_node_features_and_labels_for_each_function(test_cfg, le, max_length)


    accuracy_scores = []
    error_count = 0
    count_correct_vul = 0
    for i, test_node_features_batch in enumerate(test_node_features):
        test_node_features_batch = np.array(test_node_features_batch, dtype=float)
        test_node_label = np.array(test_node_labels[i], dtype=int)

        test_dataset = TensorDataset(torch.tensor(test_node_features_batch).float(),
                                     torch.tensor(test_node_label).float())

        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        test_predictions, test_true_labels = evaluate_model_test(clf_node, test_loader)
        node_pred_test = test_predictions
        node_pred_test = convert_to_binary_labels(node_pred_test)
        correct = has_matching_index(node_pred_test, test_node_label)
        if correct:
            count_correct_vul += 1
        accuracy = accuracy_score(test_node_labels[i], node_pred_test)
        accuracy_scores.append(accuracy)

        # print("error count: ",error_count)

    # Calculate and print the average accuracy
    average_accuracy = np.mean(accuracy_scores)
    print("Test Accuracy in node classification:", average_accuracy)
    print("Test Accuracy in localization vulnerable nodes:", count_correct_vul / len(test_node_labels))

    print("Accuracy for all correct predict bad function in testing data:")
    # print(max(correct_test_indices))
    # print(len(test_df))
    test_cfg, test_node_labels, test_graph_labels, vul_node_count, non_vul_node_count, total_node_count = create_cfg_for_node(test_df)
    # print(len(test_cfg))
    # print(len(correct_test_indices))
    correct_test_cfgs = [test_cfg[i] for i in correct_test_indices]
    test_node_features, test_node_labels = extract_node_features_and_labels_for_each_function(correct_test_cfgs, le, max_length)
    accuracy_scores = []
    error_count = 0
    count_correct_vul = 0
    for i, test_node_features_batch in enumerate(test_node_features):
        test_node_features_batch = np.array(test_node_features_batch, dtype=float)
        test_node_label = np.array(test_node_labels[i], dtype=int)
        test_dataset = TensorDataset(torch.tensor(test_node_features_batch).float(),
                                     torch.tensor(test_node_label).float())
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        test_predictions, test_true_labels = evaluate_model_test(clf_node, test_loader)
        node_pred_test = test_predictions
        node_pred_test = convert_to_binary_labels(node_pred_test)
        correct = has_matching_index(node_pred_test, test_node_label)
        if correct:
            count_correct_vul += 1
        accuracy = accuracy_score(test_node_labels[i], node_pred_test)
        accuracy_scores.append(accuracy)

        # print("error count: ", error_count)

    # Calculate and print the average accuracy
    average_accuracy = np.mean(accuracy_scores)
    print("Test Accuracy in node classification:", average_accuracy)
    print("Test Accuracy in localization vulnerable nodes:", count_correct_vul / len(test_node_labels))

def has_element_one(input_list):
    """
    Check if the input list contains the element 1.

    Args:
    input_list (list): The list to check.

    Returns:
    bool: True if the list contains 1, False otherwise.
    """
    return 1 if 1 in input_list else 0

def test_node_for_functions(asembly_length, clf_node):
    train_file = '../data/big-vul_dataset/train_graph.csv'
    val_file = '../data/big-vul_dataset/val_graph.csv'
    test_file = '../data/big-vul_dataset/test_graph.csv'

    train_df, val_df, test_df = load_data(train_file, val_file, test_file)
    le = LabelEncoder()
    le.fit(train_df['CWE ID'].tolist() + val_df['CWE ID'].tolist() + test_df['CWE ID'].tolist())

    train_df['CWE ID Encoded'] = le.transform(train_df['CWE ID'])
    val_df['CWE ID Encoded'] = le.transform(val_df['CWE ID'])
    test_df['CWE ID Encoded'] = le.transform(test_df['CWE ID'])

    test_cfg, test_node_labels, test_graph_labels,test_vul_nodes_count, test_non_vul_nodes_count, test_total_nodes_count = create_cfg_for_graph(test_df)

    print("Number cfg (both good and bad function) in testing data:", len(test_cfg))

    # with open('./saved_kernel/svm_model_node.pkl', 'rb') as f:
    #     clf_node = pickle.load(f)
    tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
    test_node_features, test_node_labels = extract_node_features_and_labels_for_each_function(test_cfg, le)
    print("Number node labels in all cfg in testing data:", sum(len(test_node_labels) for sublist in test_node_labels))
    predict_graph_labels = []

    for i, test_node_features_batch in enumerate(test_node_features):
        test_node_features_flat = np.array(test_node_features_batch, dtype=float)
        node_pred_test = clf_node.predict(test_node_features_flat) # Use the last trained model for prediction
        predict_graph_labels.append(has_element_one(node_pred_test))

    print("Test Accuracy in function classification:", accuracy_score(test_graph_labels, predict_graph_labels))

def train_node_for_functions(asembly_length):
    train_file = '../data/big-vul_dataset/train_graph.csv'
    val_file = '../data/big-vul_dataset/val_graph.csv'
    test_file = '../data/big-vul_dataset/test_graph.csv'

    train_df, val_df, test_df = load_data(train_file, val_file, test_file)
    le = LabelEncoder()
    le.fit(train_df['CWE ID'].tolist() + val_df['CWE ID'].tolist() + test_df['CWE ID'].tolist())

    train_df['CWE ID Encoded'] = le.transform(train_df['CWE ID'])
    val_df['CWE ID Encoded'] = le.transform(val_df['CWE ID'])
    test_df['CWE ID Encoded'] = le.transform(test_df['CWE ID'])

    train_cfg, train_node_labels, train_graph_labels, train_vul_nodes_count, train_non_vul_nodes_count, train_total_nodes_count = create_cfg_for_graph(
        train_df)
    print("Number node labels in testing data:", len(train_node_labels))
    print("Number cfg in testing data:", len(train_cfg))

    with open('./saved_kernel/svm_model_node.pkl', 'rb') as f:
        clf_node = pickle.load(f)
    tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
    train_node_features, train_node_labels = extract_node_features_and_labels_for_each_function(train_cfg, le)
    predict_graph_labels = []

    for i, test_node_features_batch in enumerate(train_node_features):
        test_node_features_flat = np.array(test_node_features_batch, dtype=float)
        node_pred_test = clf_node.predict(test_node_features_flat) # Use the last trained model for prediction
        predict_graph_labels.append(has_element_one(node_pred_test))

    print("Test Accuracy for vulnerable nodes and non-vulnerable nodes of training set:", accuracy_score(train_graph_labels, predict_graph_labels))
    return predict_graph_labels

def main():
    asembly_length, clf_node, correct_test_indices, max_length, le = train_all()
    test_node_top_10(asembly_length, clf_node, correct_test_indices, max_length, le)
    # test_node_for_functions(103, clf_node)


if __name__ == "__main__":
    main()