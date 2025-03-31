import pandas as pd
import networkx as nx
from sklearn.preprocessing import LabelEncoder
import json
import logging
from transformers import (RobertaModel, BertModel, RobertaForSequenceClassification, RobertaTokenizer)
import re
import kernels as kernels
import time
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Set random seed for reproducibility
random_state = 42
# tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
# model = RobertaModel.from_pretrained('microsoft/codebert-base')
# model.eval()  # Tắt chế độ training để giảm tài nguyên
with open('all_opcodes.json', 'r') as f:
        ALL_OPCODES = json.load(f)

def extract_all_opcodes(file_paths):
    opcode_set = set()
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            for cfg_field in ['processed_func_cfg', 'vul_func_with_fix_cfg']:
                if cfg_field in row and pd.notna(row[cfg_field]):
                    cfg_data = json.loads(row[cfg_field])
                    for block in cfg_data['blocks']:
                        assembly = block.get('assembly', [])
                        for line in assembly:
                            match = re.match(r"\s*0x[0-9a-fA-F]+:\s*(\w+)", line)
                            if match:
                                opcode = match.group(1)
                                opcode_set.add(opcode)
    return list(opcode_set)


def load_data(train_file, val_file, test_file):
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)
    return train_df, val_df, test_df
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

# def embed_assembly_code(assembly_code):
#     inputs = tokenizer(assembly_code, return_tensors="pt", truncation=True, max_length=512)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     # Lấy embedding của token [CLS] (vector đại diện cho toàn câu)
#     cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
#     return cls_embedding

def one_hot_opcode_vector(assembly_code):
    vector = [0] * len(ALL_OPCODES)
    for opcode in assembly_code.split('\n'):
        if opcode in ALL_OPCODES:
            index = ALL_OPCODES.index(opcode)
            vector[index] = 1
    return vector

def create_cfg_for_graph(df):
    cfg_list = []
    node_labels = []
    graph_labels = []
    vul_node_count = 0
    non_vul_node_count = 0
    total_node_count = 0
    total_edge_count = 0

    for _, row in df.iterrows():
        try:
            # Process vulnerable function CFG
            if 'processed_func_cfg' in row and pd.notna(row['processed_func_cfg']):
                cfg_data = json.loads(row['processed_func_cfg'])
                flaw_line_address = row['flaw_line_address']
                cwe_id = row['CWE ID']
                graph_label = 1  # Vulnerable graph

                cfg = nx.DiGraph()
                for block in cfg_data['blocks']:
                    address = block.get('address', None)
                    if address is not None:
                        assembly = block.get('assembly', [])
                        assembly_code = extract_opcodes(assembly)
                        
                        node_label = f"{assembly_code}"

                        node_vector = one_hot_opcode_vector(assembly_code)
                        node_vector_str = ' '.join(map(str, node_vector))

                        cfg.add_node(address, label=node_vector_str, assembly=assembly)
                        # cfg.add_node(address, label=node_label, assembly=assembly)
                        vul_node_count += 1
                        total_node_count += 1
                    else:
                        logging.warning(f"Block without address found: {block}")

                for edge in cfg_data['edges']:
                    from_node = edge['from']
                    to_node = edge['to']
                    instruction_type = 'Unknown'
                    branch_condition = 'None'
                    if 'stmt' in edge:
                        stmt = edge['stmt']
                        if stmt.is_a_jump:
                            instruction_type = 'jmp'
                        elif stmt.is_a_call:
                            instruction_type = 'call'
                        elif stmt.is_a_ret:
                            instruction_type = 'ret'
                        if stmt.is_a_conditional_jump:
                            branch_condition = stmt.condition

                    edge_label = f"Instr: {instruction_type}\nCond: {branch_condition}"
                    cfg.add_edge(from_node, to_node, label=edge_label)
                    total_edge_count += 1

                for node in cfg.nodes():
                    if 'label' not in cfg.nodes[node]:
                        cfg.nodes[node]['label'] = " "
                    if 'assembly' not in cfg.nodes[node]:
                        cfg.nodes[node]['assembly'] = []

                cfg_list.append(cfg)
                graph_labels.append(graph_label)

            # Process non-vulnerable function CFG
            if 'vul_func_with_fix_cfg' in row and pd.notna(row['vul_func_with_fix_cfg']):
                cfg_data = json.loads(row['vul_func_with_fix_cfg'])
                line_after_address = row['line_after_address']
                cwe_id = row['CWE ID']
                graph_label = 0  # Non-vulnerable graph

                cfg = nx.DiGraph()
                for block in cfg_data['blocks']:
                    address = block.get('address', None)
                    if address is not None:
                        assembly = block.get('assembly', [])
                        assembly_code = extract_opcodes(assembly)
                       
                        
                        node_label = f"{assembly_code}"

                        node_vector = one_hot_opcode_vector(assembly_code)
                        node_vector_str = ' '.join(map(str, node_vector))

                        cfg.add_node(address, label=node_vector_str, assembly=assembly)
                        # cfg.add_node(address, label=node_label, assembly=assembly)
                        non_vul_node_count += 1
                        total_node_count += 1
                    else:
                        logging.warning(f"Block without address found: {block}")

                for edge in cfg_data['edges']:
                    from_node = edge['from']
                    to_node = edge['to']
                    instruction_type = 'Unknown'
                    branch_condition = 'None'
                    if 'stmt' in edge:
                        stmt = edge['stmt']
                        if stmt.is_a_jump:
                            instruction_type = 'jmp'
                        elif stmt.is_a_call:
                            instruction_type = 'call'
                        elif stmt.is_a_ret:
                            instruction_type = 'ret'
                        if stmt.is_a_conditional_jump:
                            branch_condition = stmt.condition

                    edge_label = f"Instr: {instruction_type}\nCond: {branch_condition}"
                    cfg.add_edge(from_node, to_node, label=edge_label)
                    total_edge_count += 1

                for node in cfg.nodes():
                    if 'label' not in cfg.nodes[node]:
                        cfg.nodes[node]['label'] = " "
                    if 'assembly' not in cfg.nodes[node]:
                        cfg.nodes[node]['assembly'] = []
                    total_node_count += 1

                cfg_list.append(cfg)
                graph_labels.append(graph_label)

        except Exception as e:
            logging.error(f"Error processing row: {e}")

    return cfg_list, node_labels, graph_labels


def train_graph_classifier(train_df, val_df, test_df):
    # predict_graph_labels_node_classifier = train_node_for_functions(asembly_length)
    ful_file = '../data/big-vul_dataset/test_13_target_1_not_null.csv'


    train_cfg, train_node_labels, train_graph_labels = create_cfg_for_graph(
        train_df)
    print("Number node labels in training data:", len(train_node_labels))
    print("Number cfg in training data:", len(train_cfg))
    val_cfg, val_node_labels, val_graph_labels = create_cfg_for_graph(
        val_df)
    print("Number node labels in validating data:", len(val_node_labels))
    print("Number cfg in validating data:", len(val_cfg))
    test_cfg, test_node_labels, test_graph_labels = create_cfg_for_graph(
        test_df)
    print("Number node labels in testing data:", len(test_node_labels))
    print("Number cfg in testing data:", len(test_cfg))

    batch_size = 10000
    # Start the timer
    start_time = time.time()
    
    correct_test_indices = kernels.wl(train_cfg,train_graph_labels, batch_size, test_graph_labels, test_cfg)
    # correct_test_indices = kernels.shortest_path_kernel(train_cfg, train_graph_labels, batch_size, test_graph_labels, test_cfg)
    # correct_test_indices = kernels.subgraph(train_cfg, train_graph_labels, batch_size, test_graph_labels, test_cfg)
    # correct_test_indices = kernels.wl_oa(train_cfg, train_graph_labels, batch_size, test_graph_labels, test_cfg)
    # correct_test_indices = kernels.wl_oa_sp(train_cfg, train_graph_labels, batch_size, test_graph_labels, test_cfg)
    # correct_test_indices = kernels.wl_oa_sp_pca(train_cfg, train_graph_labels, batch_size, test_graph_labels, test_cfg)
    # correct_test_indices = kernels.wl_oa_rbf(train_cfg, train_graph_labels, batch_size, test_graph_labels, test_cfg)
    # End the timer

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    return correct_test_indices

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
    correct_test_indices = train_graph_classifier(train_df, val_df, test_df)
    # assembly_length, clf_node = train_node_classifier(train_df, val_df, test_df, le)

def main():

   
    
    
     train_all()
    # test_node_top_10(103, clf_graph, gk)
    # test_node_for_functions(103, clf_node)


if __name__ == "__main__":
    main()