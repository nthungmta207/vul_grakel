import pandas as pd
import networkx as nx
from grakel import Graph, GraphKernel
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import pickle
import json
import logging
import numpy as np
import node_models as node_models
from torch.utils.data import DataLoader, TensorDataset
import torch

from transformers import (RobertaModel, BertModel, RobertaForSequenceClassification, RobertaTokenizer)

import re


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Set random seed for reproducibility

tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
# model = RobertaModel.from_pretrained('roberta-base')

def load_data(train_file, val_file, test_file):
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)
    return train_df, val_df, test_df


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

def merge_adjacent_nodes(cfg):
    merged_cfg = nx.DiGraph()
    merged_node_map = {}  # Map từ (u, v) -> id mới để tránh trùng

    for u, v in cfg.edges():
        node_u = cfg.nodes[u]
        node_v = cfg.nodes[v]
        
        # Gộp assembly
        merged_assembly = node_u.get('assembly', []) + node_v.get('assembly', [])
        
        # Gộp label (tùy mục đích, có thể nối chuỗi hoặc tính toán khác)
        label_u = node_u.get('label', '')
        label_v = node_v.get('label', '')
        merged_label = f"{label_u}|{label_v}"

        merged_node_id = (u, v)  # Dùng tuple làm ID node mới
        merged_cfg.add_node(merged_node_id, assembly=merged_assembly, label=merged_label)

        merged_node_map[(u, v)] = merged_node_id

    # Thêm cạnh giữa các node ghép nếu tồn tại chuỗi liên tiếp
    for (u1, v1), merged_node_1 in merged_node_map.items():
        for (u2, v2), merged_node_2 in merged_node_map.items():
            if v1 == u2:
                merged_cfg.add_edge(merged_node_1, merged_node_2)

    return merged_cfg


def extract_features_for_merged_nodes(cfg_list, le):
    merged_node_features = []
    merged_node_labels = []

    for merged_node, attributes in cfg.nodes(data=True):
        (u, v) = merged_node
        node_u = cfg.nodes.get(u, {})
        node_v = cfg.nodes.get(v, {})

        assembly_u = node_u.get('assembly', [])
        assembly_v = node_v.get('assembly', [])
        merged_assembly = assembly_u + assembly_v
        assembly_code = extract_opcode_operand(merged_assembly)
        assembly_list = assembly_code.split()

        label_u = node_u.get('label', '0_0')
        label_v = node_v.get('label', '0_0')
        cwe_u, is_vul_u = label_u.split('_')
        cwe_v, is_vul_v = label_v.split('_')

        features = {
            "length": len(assembly_list),
            "registers_used": len([word for word in assembly_list if word.startswith('e')]),
            "constants": len([word for word in assembly_list if word.isdigit()]),
            "is_jump": sum(' '.join(assembly_list).count(jump) for jump in ["jmp", "je", "jne", "jg", "jl"]),
            "memory_address": sum(line.count('[') for line in assembly_list),
            "is_function_call": int("call" in ' '.join(assembly_list)),
            "in_degree": cfg.in_degree(merged_node),
            "out_degree": cfg.out_degree(merged_node),
            "is_entry_point": int(cfg.in_degree(merged_node) == 0),
            "is_exit_point": int(cfg.out_degree(merged_node) == 0)
        }

        merged_node_features.append(list(features.values()))
        merged_node_labels.append(int(is_vul_u == '1' or is_vul_v == '1'))

    print(f"Extracted {len(merged_node_features)} merged node features.")
    return merged_node_features, merged_node_labels

def extract_features_for_merged_nodes_each_funtion(cfg, le):
    merged_node_features = []
    merged_node_labels = []

    for merged_node, attributes in cfg.nodes(data=True):
        node_features = []
        node_labels = []
        (u, v) = merged_node
        node_u = cfg.nodes.get(u, {})
        node_v = cfg.nodes.get(v, {})

        assembly_u = node_u.get('assembly', [])
        assembly_v = node_v.get('assembly', [])
        merged_assembly = assembly_u + assembly_v
        assembly_code = extract_opcode_operand(merged_assembly)
        assembly_list = assembly_code.split()

        label_u = node_u.get('label', '0_0')
        label_v = node_v.get('label', '0_0')
        cwe_u, is_vul_u = label_u.split('_')
        cwe_v, is_vul_v = label_v.split('_')

        features = {
            "length": len(assembly_list),
            "registers_used": len([word for word in assembly_list if word.startswith('e')]),
            "constants": len([word for word in assembly_list if word.isdigit()]),
            "is_jump": sum(' '.join(assembly_list).count(jump) for jump in ["jmp", "je", "jne", "jg", "jl"]),
            "memory_address": sum(line.count('[') for line in assembly_list),
            "is_function_call": int("call" in ' '.join(assembly_list)),
            "in_degree": cfg.in_degree(merged_node),
            "out_degree": cfg.out_degree(merged_node),
            "is_entry_point": int(cfg.in_degree(merged_node) == 0),
            "is_exit_point": int(cfg.out_degree(merged_node) == 0)
        }

        node_features.append(list(features.values()))
        node_labels.append(int(is_vul_u == '1' or is_vul_v == '1'))

    print(f"Extracted {len(merged_node_features)} merged node features.")
    return merged_node_features, merged_node_labels

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

                cfg = merge_adjacent_nodes(cfg)
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

                cfg = merge_adjacent_nodes(cfg)
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
                cfg = merge_adjacent_nodes(cfg)
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

def extract_node_features_and_labels(cfg_list, le):
    node_features = []
    node_labels = []
    count_vul = 0
    count_non_vul = 0

    for cfg in cfg_list:
        node_addresses = []
        for node in cfg.nodes(data=True):
            address, attributes = node
            cwe_id, is_vulnerable = attributes['label'].split('_')
            if is_vulnerable == '0' or is_vulnerable == '1':
                cwe_id_encoded = le.transform([cwe_id])[0]
                assembly = attributes.get('assembly', [])
                assembly_code = extract_opcode_operand(assembly)
                assembly_list = assembly_code.split()
                # Trích xuất các features
                features = {
                    "length": len(assembly_list),
                    "instruction_type": ord(assembly_code[0].split()[0][0]) if assembly_list else 0,
                    "registers_used": len([word for word in assembly_list if word.startswith('e')]),
                    "constants": len([word for word in assembly_list if word.isdigit()]),
                    # "is_jump": int(any(jump in ' '.join(assembly_list) for jump in ["jmp", "je", "jne", "jg", "jl"])),
                    # "memory_address": int("[" in ' '.join(assembly_code)),
                    "is_jump": sum(' '.join(assembly_list).count(jump) for jump in ["jmp", "je", "jne", "jg", "jl"]),
                    "memory_address": sum(line.count('[') for line in assembly_list),
                    "is_function_call": int("call" in ' '.join(assembly_list)),
                    "in_degree": cfg.in_degree(address),
                    "out_degree": cfg.out_degree(address),
                    "is_entry_point": int(cfg.in_degree(address) == 0),
                    "is_exit_point": int(cfg.out_degree(address) == 0)
                }

                # Thêm features vào danh sách node_features
                node_features.append(list(features.values()))
                node_labels.append(int(is_vulnerable))
                if is_vulnerable == '1':
                    count_vul += 1
                else:
                    count_non_vul += 1
    print("Number of vulnerable nodes after extraction:", count_vul)
    print("Number of non-vulnerable nodes after extraction:", count_non_vul)
    return node_features, node_labels
def extract_node_features_and_labels_for_each_function(cfg_list, le):
    all_node_features = []
    all_node_labels = []

    for cfg in cfg_list:
        node_features = []
        node_labels = []
        for node in cfg.nodes(data=True):
            address, attributes = node
            cwe_id, is_vulnerable = attributes['label'].split('_')
            cwe_id_encoded = le.transform([cwe_id])[0]
            assembly = attributes.get('assembly', [])
            assembly_code = extract_opcode_operand(assembly)
            assembly_list = assembly_code.split()
            # Trích xuất các features
            features = {
                "length": len(assembly_list),
                "instruction_type": ord(assembly_code[0].split()[0][0]) if assembly_list else 0,
                "registers_used": len([word for word in assembly_list if word.startswith('e')]),
                "constants": len([word for word in assembly_list if word.isdigit()]),
                # "is_jump": int(any(jump in ' '.join(assembly_list) for jump in ["jmp", "je", "jne", "jg", "jl"])),
                # "memory_address": int("[" in ' '.join(assembly_code)),
                "is_jump": sum(' '.join(assembly_list).count(jump) for jump in ["jmp", "je", "jne", "jg", "jl"]),
                "memory_address": sum(line.count('[') for line in assembly_list),
                "is_function_call": int("call" in ' '.join(assembly_list)),
                "in_degree": cfg.in_degree(address),
                "out_degree": cfg.out_degree(address),
                "is_entry_point": int(cfg.in_degree(address) == 0),
                "is_exit_point": int(cfg.out_degree(address) == 0)
            }
            # Thêm features vào danh sách node_features
            node_features.append(list(features.values()))
            node_labels.append(int(is_vulnerable))

        all_node_features.append(node_features)
        all_node_labels.append(node_labels)

    return all_node_features, all_node_labels
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


def train_all():
    train_file = '../data/big-vul_dataset/train_graph.csv'
    val_file = '../data/big-vul_dataset/val_graph.csv'
    test_file = '../data/big-vul_dataset/test_graph.csv'

    train_df, val_df, test_df = load_data(train_file, val_file, test_file)

    le = LabelEncoder()
    le.fit(train_df['CWE ID'].tolist() + val_df['CWE ID'].tolist() + test_df['CWE ID'].tolist())

    train_df['CWE ID Encoded'] = le.transform(train_df['CWE ID'])
    val_df['CWE ID Encoded'] = le.transform(val_df['CWE ID'])
    test_df['CWE ID Encoded'] = le.transform(test_df['CWE ID'])
    # train_graph_classifier(train_df, val_df, test_df)
    assembly_length, clf_node, correct_test_indices = node_models.randon_forest(train_df, val_df, test_df, le)

    return assembly_length, clf_node, correct_test_indices

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
def test_node_top_10(asembly_length, clf_node,correct_test_indices ):
    train_file = '../data/big-vul_dataset/train_graph.csv'
    val_file = '../data/big-vul_dataset/val_graph.csv'
    test_file = '../data/big-vul_dataset/test_graph.csv'

    train_df, val_df, test_df = load_data(train_file, val_file, test_file)

    le = LabelEncoder()
    le.fit(train_df['CWE ID'].tolist() + val_df['CWE ID'].tolist() + test_df['CWE ID'].tolist())

    train_df['CWE ID Encoded'] = le.transform(train_df['CWE ID'])
    val_df['CWE ID Encoded'] = le.transform(val_df['CWE ID'])
    test_df['CWE ID Encoded'] = le.transform(test_df['CWE ID'])

    test_cfg, test_node_labels, test_graph_labels = create_cfg_for_top_k(test_df)



    print("Number node labels in all bad fucntion in testing data:", len(test_node_labels))
    print("Number cfg (bad function only) in testing data:", len(test_cfg))

    print("Unique node labels in testing data:", set([label for _, label in test_node_labels]))
    print("Count of each node label in testing data:",
          {label: sum(1 for _, l in test_node_labels if l == label) for label in
           set([label for _, label in test_node_labels])})

    # with open('./saved_kernel/svm_model_node.pkl', 'rb') as f:
    #     clf_node = pickle.load(f)

    print("Accuracy for all bad function in testing data:")
    test_node_features, test_node_labels = extract_node_features_and_labels_for_each_function(test_cfg, le)
    accuracy_scores = []
    count_correct_vul = 0
    for i, test_node_features_batch in enumerate(test_node_features):
        test_node_features_flat = np.array(test_node_features_batch, dtype=float)
        node_pred_test = clf_node.predict_proba(test_node_features_flat)[:,
                         1]  # Use the last trained model for prediction
        node_pred_test = convert_to_binary_labels(node_pred_test)
        correct = has_matching_index(node_pred_test, test_node_labels[i])
        if correct:
            count_correct_vul += 1
        accuracy = accuracy_score(test_node_labels[i], node_pred_test)
        accuracy_scores.append(accuracy)
        # print(classification_report(test_node_labels, node_pred_test, digits=5))

    # Calculate and print the average accuracy
    average_accuracy = np.mean(accuracy_scores)
    print("Test Accuracy in node classification:", average_accuracy)
    print("Test Accuracy in localization vulnerable nodes:", count_correct_vul / len(test_node_labels))

    print("Accuracy for all correct predict bad function in testing data:")
    print(max(correct_test_indices))
    print(len(test_df))
    test_cfg, test_node_labels, test_graph_labels, vul_node_count, non_vul_node_count, total_node_count = create_cfg_for_node(test_df)
    # print(len(test_cfg))
    # print(len(correct_test_indices))
    correct_test_cfgs = [test_cfg[i] for i in correct_test_indices]
    test_node_features, test_node_labels = extract_node_features_and_labels_for_each_function(correct_test_cfgs, le)
    accuracy_scores = []
    count_correct_vul = 0
    for i, test_node_features_batch in enumerate(test_node_features):
        test_node_features_flat = np.array(test_node_features_batch, dtype=float)
        node_pred_test = clf_node.predict_proba(test_node_features_flat)[:,
                         1]  # Use the last trained model for prediction
        node_pred_test = convert_to_binary_labels(node_pred_test)
        correct = has_matching_index(node_pred_test, test_node_labels[i])
        if correct:
            count_correct_vul += 1
        accuracy = accuracy_score(test_node_labels[i], node_pred_test)
        accuracy_scores.append(accuracy)
        # print(classification_report(test_node_labels, node_pred_test, digits=5))

    # Calculate and print the average accuracy
    average_accuracy = np.mean(accuracy_scores)
    print("Test Accuracy in node classification:", average_accuracy)
    print("Test Accuracy in localization vulnerable nodes:", count_correct_vul / len(test_node_labels))

def test_node_top_10_cnn(asembly_length, clf_node,correct_test_indices ):
    train_file = '../data/big-vul_dataset/train_graph.csv'
    val_file = '../data/big-vul_dataset/val_graph.csv'
    test_file = '../data/big-vul_dataset/test_graph.csv'

    train_df, val_df, test_df = load_data(train_file, val_file, test_file)

    le = LabelEncoder()
    le.fit(train_df['CWE ID'].tolist() + val_df['CWE ID'].tolist() + test_df['CWE ID'].tolist())

    train_df['CWE ID Encoded'] = le.transform(train_df['CWE ID'])
    val_df['CWE ID Encoded'] = le.transform(val_df['CWE ID'])
    test_df['CWE ID Encoded'] = le.transform(test_df['CWE ID'])

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
    test_node_features, test_node_labels = extract_node_features_and_labels_for_each_function(test_cfg, le)


    accuracy_scores = []
    error_count = 0
    count_correct_vul = 0
    for i, test_node_features_batch in enumerate(test_node_features):
        test_node_features_batch = np.array(test_node_features_batch, dtype=float)
        test_node_label = np.array(test_node_labels[i], dtype=int)
        test_dataset = TensorDataset(torch.tensor(test_node_features_batch).float(), torch.tensor(test_node_label).float())
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        if len(test_loader) == 1:
            test_predictions, test_true_labels = node_models.evaluate_model_test_cnn(clf_node, test_loader)
            node_pred_test = test_predictions
            node_pred_test = convert_to_binary_labels(node_pred_test)
            correct = has_matching_index(node_pred_test, test_node_label)
            if correct:
                count_correct_vul += 1
            accuracy = accuracy_score(test_node_labels[i], node_pred_test)
            accuracy_scores.append(accuracy)
        else:
            error_count += 1
        # print("error count: ",error_count)

    # Calculate and print the average accuracy
    average_accuracy = np.mean(accuracy_scores)
    print("Test Accuracy in node classification:", average_accuracy)
    print("Test Accuracy in localization vulnerable nodes:", count_correct_vul / len(test_node_labels))

    print("Accuracy for all correct predict bad function in testing data:")
    print(max(correct_test_indices))
    print(len(test_df))
    test_cfg, test_node_labels, test_graph_labels, vul_node_count, non_vul_node_count, total_node_count = create_cfg_for_node(test_df)
    # print(len(test_cfg))
    # print(len(correct_test_indices))
    correct_test_cfgs = [test_cfg[i] for i in correct_test_indices]
    test_node_features, test_node_labels = extract_node_features_and_labels_for_each_function(correct_test_cfgs, le)
    accuracy_scores = []
    error_count = 0
    count_correct_vul = 0
    for i, test_node_features_batch in enumerate(test_node_features):
        test_node_features_batch = np.array(test_node_features_batch, dtype=float)
        test_node_label = np.array(test_node_labels[i], dtype=int)
        test_dataset = TensorDataset(torch.tensor(test_node_features_batch).float(),
                                     torch.tensor(test_node_label).float())
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        if len(test_loader) == 1:
            test_predictions, test_true_labels = node_models.evaluate_model_test_cnn(clf_node, test_loader)
            node_pred_test = test_predictions
            node_pred_test = convert_to_binary_labels(node_pred_test)
            correct = has_matching_index(node_pred_test, test_node_label)
            if correct:
                count_correct_vul += 1
            accuracy = accuracy_score(test_node_labels[i], node_pred_test)
            accuracy_scores.append(accuracy)
        else:
            error_count += 1
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
    asembly_length, clf_node, correct_test_indices = train_all()

    test_node_top_10(asembly_length, clf_node, correct_test_indices)
    # test_node_for_functions(asembly_length, clf_node)


if __name__ == "__main__":
    main()