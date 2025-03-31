import pandas as pd
import networkx as nx
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import json
import logging
import re
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

random_state = 42

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

def create_graph_features(cfg):
    """Extract meaningful features from a CFG graph."""
    num_nodes = cfg.number_of_nodes()
    num_edges = cfg.number_of_edges()
    avg_degree = sum(dict(cfg.degree()).values()) / num_nodes if num_nodes > 0 else 0
    density = nx.density(cfg)
    clustering_coeff = nx.average_clustering(cfg.to_undirected()) if num_nodes > 1 else 0
    max_degree = max(dict(cfg.degree()).values(), default=0)
    
    return [num_nodes, num_edges, avg_degree, density, clustering_coeff, max_degree]

def create_cfg_for_graph(df):
    graph_features = []
    graph_labels = []

    registers_list = ["eax", "ebx", "ecx", "edx", "esi", "edi", "esp", "ebp"]

    for _, row in df.iterrows():
        try:
            if 'processed_func_cfg' in row and pd.notna(row['processed_func_cfg']):
                cfg_data = json.loads(row['processed_func_cfg'])
                graph_label = 1  # Vulnerable graph
                cfg = nx.DiGraph()
                
                total_jump_count = 0
                total_memory_access_count = 0
                total_registers_used = 0
                total_function_call = 0


                for block in cfg_data['blocks']:
                    address = block.get('address', None)
                    if address is not None:
                        assembly = block.get('assembly', [])
                        assembly_code = extract_opcodes(assembly)
                        cfg.add_node(address, label=assembly_code)

                        # Đếm số lần jump và memory access
                        block_code = ' '.join(assembly)
                        total_jump_count += sum(block_code.count(jump) for jump in ["jmp", "je", "jne", "jg", "jl"])
                        total_memory_access_count += block_code.count('[')
                        total_registers_used += sum(block_code.count(reg) for reg in registers_list)
                        total_function_call += block_code.count('call')

                for edge in cfg_data['edges']:
                    from_node = edge['from']
                    to_node = edge['to']
                    cfg.add_edge(from_node, to_node)

                graph_feature = create_graph_features(cfg)
                graph_feature.append(total_jump_count)
                graph_feature.append(total_memory_access_count)
                graph_feature.append(total_registers_used)
                graph_feature.append(total_function_call)
                
                graph_features.append(graph_feature)
                graph_labels.append(graph_label)
            if 'vul_func_with_fix_cfg' in row and pd.notna(row['vul_func_with_fix_cfg']):
                cfg_data = json.loads(row['vul_func_with_fix_cfg'])
                graph_label = 0  # Non-vulnerable graph
                cfg = nx.DiGraph()
                total_jump_count = 0
                total_memory_access_count = 0
                total_registers_used = 0
                total_function_call = 0

                for block in cfg_data['blocks']:
                    address = block.get('address', None)
                    if address is not None:
                        assembly = block.get('assembly', [])
                        assembly_code = extract_opcodes(assembly)
                        cfg.add_node(address, label=assembly_code)
                        
                        # Đếm số lần jump và memory access
                        block_code = ' '.join(assembly)
                        total_jump_count += sum(block_code.count(jump) for jump in ["jmp", "je", "jne", "jg", "jl"])
                        total_memory_access_count += block_code.count('[')
                        total_registers_used += sum(block_code.count(reg) for reg in registers_list)
                        total_function_call += block_code.count('call')

                for edge in cfg_data['edges']:
                    from_node = edge['from']
                    to_node = edge['to']
                    cfg.add_edge(from_node, to_node)

                graph_feature = create_graph_features(cfg)
                graph_feature.append(total_jump_count)
                graph_feature.append(total_memory_access_count)
                graph_feature.append(total_registers_used)
                graph_feature.append(total_function_call)
                
                graph_features.append(graph_feature)
                graph_labels.append(graph_label)

           

        except Exception as e:
            logging.error(f"Error processing row: {e}")

    return graph_features, graph_labels

def optimize_hyperparameters(train_features, train_labels):
    """Use GridSearchCV to find the best hyperparameters."""
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
    }

    rf = RandomForestClassifier(random_state=random_state)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)

    grid_search.fit(train_features, train_labels)
    print(f"Best Parameters: {grid_search.best_params_}")
    
    return grid_search.best_estimator_

def train_random_forest(train_features, train_labels, val_features, val_labels, test_features, test_labels):
    rf = optimize_hyperparameters(train_features, train_labels)

    # Validate the model
    val_predictions = rf.predict(val_features)
    val_accuracy = accuracy_score(val_labels, val_predictions)
    print(f"Validation Accuracy: {val_accuracy:.2f}")
    print(classification_report(val_labels, val_predictions))

    # Test the model
    test_predictions = rf.predict(test_features)
    test_accuracy = accuracy_score(test_labels, test_predictions)
    print(f"Test Accuracy: {test_accuracy:.2f}")
    print(classification_report(test_labels, test_predictions))

    return rf

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

    train_features, train_labels = create_cfg_for_graph(train_df)
    val_features, val_labels = create_cfg_for_graph(val_df)
    test_features, test_labels = create_cfg_for_graph(test_df)

    train_random_forest(train_features, train_labels, val_features, val_labels, test_features, test_labels)

def main():
    start_time = time.time()
    train_all()
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
