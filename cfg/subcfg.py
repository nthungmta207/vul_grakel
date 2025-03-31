import pandas as pd
import networkx as nx
from grakel import Graph, GraphKernel
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer
import json
import logging
import re
import numpy as np
import kernels as kernels
from joblib import Parallel, delayed
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from itertools import combinations

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
random_state = 42
tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')


def load_data(train_file, val_file, test_file):
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)
    return train_df, val_df, test_df


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

def process_in_batches(data, labels, batch_size):
    num_batches = (len(data) + batch_size - 1) // batch_size
    print(f"Processing data in {num_batches} batches")
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size], labels[i:i + batch_size]
def create_cfg_for_graph(df, le, tokenizer):
    cfg_list = []
    node_labels = []
    graph_labels = []
    node_addresses = []
    vul_node_count = 0
    non_vul_node_count = 0
    total_node_count = 0
    total_edge_count = 0
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
                            assembly = block.get('assembly', [])
                            assembly_code = extract_opcodes(assembly)
                            node_label = f"{assembly_code}"

                            cfg.add_node(address, label=node_label,
                                         assembly=block.get('assembly', []))
                            vul_node_count += 1
                            total_node_count += 1
                            node_addresses.append(address)
                        else:
                            is_vulnerable = 0
                            assembly = block.get('assembly', [])
                            assembly_code = extract_opcodes(assembly)
                            node_label = f"{assembly_code}"

                            cfg.add_node(address, label=node_label,
                                         assembly=block.get('assembly', []))
                            total_node_count += 1
                    else:
                        logging.warning(f"Block without address found: {block}")

                for edge in cfg_data['edges']:
                    cfg.add_edge(edge['from'], edge['to'])
                    total_edge_count += 1

                for node in cfg.nodes():
                    if 'label' not in cfg.nodes[node]:
                        cfg.nodes[node]['label'] = f" "
                    if 'assembly' not in cfg.nodes[node]:
                        cfg.nodes[node]['assembly'] = []

                cfg_list.append(cfg)
                graph_labels.append(graph_label)



        except Exception as e:
            logging.error(f"Error processing row: {e}")

    return cfg_list,node_addresses


def create_subcfgs(cfg_list, node_addresses, num_nodes=1):
    print("Number of nodes:", num_nodes)
    subcfgs = []
    subcfg_labels = []

    for cfg, node_address in zip(cfg_list, node_addresses):
        for node in cfg.nodes():   
            # Start with a small radius and increase until the number of nodes matches `num_nodes`
            radius = 1
            
            while radius <= num_nodes:
                # Create an ego graph (subgraph) around the node with the current radius
                if num_nodes != 1:
                    subgraph = nx.ego_graph(cfg, node, radius=radius)
                else: 
                    subgraph = nx.ego_graph(cfg, node, radius=0)
                # Check if the subgraph contains the exact number of nodes
                if len(subgraph.nodes()) == num_nodes:
                   subcfg_label = 1 if node_address in subgraph.nodes() else 0
                   subcfgs.append(subgraph)
                   subcfg_labels.append(subcfg_label)
                
                # Increase the radius and try again
                radius += 1

    return subcfgs, subcfg_labels


def convert_cfg_to_grakel(cfg):
    adjacency_matrix = nx.adjacency_matrix(cfg).todense()
    node_labels = {i: cfg.nodes[node]['label'] for i, node in enumerate(cfg.nodes)}
    return Graph(adjacency_matrix, node_labels=node_labels)


def classify_subcfgs(train_cfg, train_node_addresses, test_cfg, test_node_addresses):
    train_subcfgs, train_subcfg_labels = create_subcfgs(train_cfg, train_node_addresses)
    test_subcfgs, test_subcfg_labels = create_subcfgs(test_cfg, test_node_addresses)

    print("Unique node_labels:", set(train_subcfg_labels))
    if len(set(train_subcfg_labels)) < 2:
        raise ValueError("Training data contains only one class. Cannot train node classifier.")


    batch_size = 10000
    gk = GraphKernel(kernel=[{"name": "weisfeiler_lehman", "n_iter": 5}, {"name": "vertex_histogram"}], normalize=True)

    svc = SVC(kernel="precomputed", probability=True, random_state=42)
    clf_subcfg = GridSearchCV(svc, param_grid={'C': [0.1, 1, 10, 100, 1000]}, cv=5, n_jobs=-1)

    for i, (train_batch, train_labels_batch) in enumerate(
            process_in_batches(train_subcfgs, train_subcfg_labels, batch_size)):
        train_graphs = Parallel(n_jobs=-1)(delayed(convert_cfg_to_grakel)(cfg) for cfg in train_batch)

        K_train_batch = gk.fit_transform(train_graphs)

        train_labels_batch = np.array(train_labels_batch).flatten()

        print(
            f"Batch {i} - K_train_batch shape: {K_train_batch.shape}, train_labels_batch shape: {train_labels_batch.shape}")

        clf_subcfg.fit(K_train_batch, train_labels_batch)



    test_graphs = Parallel(n_jobs=-1)(delayed(convert_cfg_to_grakel)(cfg) for cfg in test_subcfgs)
    K_test = gk.transform(test_graphs)
    test_pred = clf_subcfg.predict(K_test)

    print("Sub-CFG Test Accuracy:", accuracy_score(test_subcfg_labels, test_pred))
    f1 = f1_score(test_subcfg_labels, test_pred, average='weighted')  # Hoặc 'macro', 'micro', tùy theo yêu cầu
    print(f"F1 Score: {f1:.5f}")
    print("Sub-CFG Test Classification Report:")
    print(classification_report(test_subcfg_labels, test_pred, digits=5))


    return clf_subcfg, gk


def evaluate_subcfg_predictions(clf, gk, test_cfg, test_node_addresses):
    total_correct = 0
    total_cfgs = len(test_cfg)
    total_subcfg_with_label_1 = 0
    total_correct_cfg = 0

    for cfg, true_node_address in zip(test_cfg, test_node_addresses):
        subcfgs, _ = create_subcfgs([cfg], [true_node_address])
        grakel_graphs = [convert_cfg_to_grakel(subcfg) for subcfg in subcfgs]

        if len(grakel_graphs) == 0:
            continue

        X_subcfg = gk.transform(grakel_graphs)
        y_pred_subcfg = clf.predict(X_subcfg)

        subcfgs_with_label_1 = [subcfg for subcfg, label in zip(subcfgs, y_pred_subcfg) if label == 1]
        if len(subcfgs_with_label_1) == 0:
            continue

        total_subcfg_with_label_1 += len(subcfgs_with_label_1)
        total_correct_cfg += 1

        node_frequency = {}
        for subcfg in subcfgs_with_label_1:
            for node in subcfg.nodes():
                if node in node_frequency:
                    node_frequency[node] += 1
                else:
                    node_frequency[node] = 1

        most_frequent_node = max(node_frequency, key=node_frequency.get)

        if most_frequent_node == true_node_address:
            total_correct += 1

    accuracy = total_correct / total_correct_cfg if total_correct_cfg > 0 else 0
    ratio = total_correct / total_subcfg_with_label_1 if total_subcfg_with_label_1 > 0 else 0

    print(f"Total Correct Predictions: {total_correct}")
    print(f"Total CFGs: {total_cfgs}")
    print(f"Total Correct CFGs: {total_correct_cfg}")
    print(f"Total Sub-CFGs with Label 1: {total_subcfg_with_label_1}")
    print(f"Accuracy: {accuracy:.5f}")
    print(f"Ratio of Correct Predictions: {ratio:.5f}")

    return accuracy, ratio

def main():
    train_file = '../data/big-vul_dataset/train_graph.csv'
    val_file = '../data/big-vul_dataset/val_graph.csv'
    test_file = '../data/big-vul_dataset/test_graph.csv'

    train_df, val_df, test_df = load_data(train_file, val_file, test_file)

    le = LabelEncoder()
    le.fit(train_df['CWE ID'].tolist() + val_df['CWE ID'].tolist() + test_df['CWE ID'].tolist())

    train_df['CWE ID Encoded'] = le.transform(train_df['CWE ID'])
    val_df['CWE ID Encoded'] = le.transform(val_df['CWE ID'])
    test_df['CWE ID Encoded'] = le.transform(test_df['CWE ID'])

    train_cfg, train_node_address = create_cfg_for_graph(train_df, le, tokenizer)
    test_cfg, test_node_address = create_cfg_for_graph(test_df, le, tokenizer)
    val_cfg, val_node_address = create_cfg_for_graph(val_df, le, tokenizer)
    # node_addresses = [list(cfg.nodes())[0] for cfg in cfg_list]  # Use the first node address of each CFG as an example

    clf_subcfg, gk = classify_subcfgs(train_cfg, train_node_address,  test_cfg, test_node_address)
    # evaluate_subcfg_predictions(clf_subcfg, gk, test_cfg, test_node_address)


if __name__ == "__main__":
    main()
