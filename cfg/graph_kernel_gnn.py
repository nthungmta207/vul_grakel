import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.preprocessing import LabelEncoder
import json
import logging
import re
from sklearn.metrics import f1_score, classification_report


torch.manual_seed(42)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def hex_to_int(hex_str):
    return int(hex_str, 16)

def extract_opcodes(assembly_lines):
    opcodes_str = ""
    for line in assembly_lines:
        match = re.match(r"\s*0x[0-9a-fA-F]+:\s*(\w+)\s*", line)
        if match:
            opcodes_str += f"{match.group(1)}\n"
        else:
            opcodes_str += "\n"
    return opcodes_str.strip()

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

# Bỏ các đặc trưng tự thiết kế, thay thế bằng việc mã hóa opcode
def create_graph_data(df, label_encoder, max_length):
    data_list = []
    for _, row in df.iterrows():
        try:
            if 'processed_func_cfg' in row and pd.notna(row['processed_func_cfg']):
                cfg_data = json.loads(row['processed_func_cfg'])
                graph_label = 1  # Đồ thị chứa lỗ hổng

                cfg = nx.DiGraph()  # Đồ thị có hướng
                node_features = []
                node_mapping = {}  # Ánh xạ từ địa chỉ gốc sang chỉ số nút mới
                next_node_index = 0  # Chỉ số tiếp theo cho các nút mới

                # Bước 1: Xử lý các nút và tạo ánh xạ cho địa chỉ sang chỉ số
                for block in cfg_data['blocks']:
                    address = block.get('address', None)
                    if address is not None:
                        assembly = block.get('assembly', [])
                        assembly_code = extract_opcodes(assembly)
                        assembly_list = assembly_code.split('\n')

                        # Mã hóa opcode thành số nguyên
                        encoded_opcodes = label_encoder.transform(assembly_list)

                        # Giới hạn độ dài của opcode hoặc padding nếu cần
                        if len(encoded_opcodes) > max_length:
                            encoded_opcodes = encoded_opcodes[:max_length]  # Cắt bớt nếu quá dài
                        else:
                            # Đệm thêm 0 nếu quá ngắn
                            padding = [0] * (max_length - len(encoded_opcodes))
                            encoded_opcodes = list(encoded_opcodes) + padding
                        # Chuyển đổi đặc trưng opcode thành tensor
                        features_tensor = torch.tensor(encoded_opcodes, dtype=torch.float)

                        node_features.append(features_tensor)

                        # Ánh xạ địa chỉ nút thành chỉ số tuần tự
                        if address not in node_mapping:
                            node_mapping[address] = next_node_index
                            next_node_index += 1

                # Bước 2: Xử lý các cạnh và sử dụng chỉ số được ánh xạ
                edge_list = []
                for edge in cfg_data['edges']:
                    from_node = edge['from']
                    to_node = edge['to']

                    if from_node not in node_mapping or to_node not in node_mapping:
                        # logging.error(f"Cạnh chứa nút không hợp lệ: {from_node} -> {to_node}")
                        continue

                    from_node_idx = node_mapping[from_node]
                    to_node_idx = node_mapping[to_node]

                    edge_list.append((from_node_idx, to_node_idx))

                edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

                if len(edge_index) == 0 or len(node_features) == 0:
                    # logging.warning("Đồ thị không có cạnh hoặc nút hợp lệ, bỏ qua.")
                    continue

            if len(node_features) > 0:
                # Chuyển đổi đặc trưng của nút thành tensor
                node_features = torch.stack(node_features)

                graph_data = Data(x=node_features, edge_index=edge_index, y=torch.tensor([graph_label], dtype=torch.float))
                data_list.append(graph_data)
                    
            if 'vul_func_with_fix_cfg' in row and pd.notna(row['vul_func_with_fix_cfg']):
                cfg_data = json.loads(row['vul_func_with_fix_cfg'])
                graph_label = 0  # Đồ thị đã sửa lỗi
                
                cfg = nx.DiGraph()  # Đồ thị có hướng
                node_features = []
                node_mapping = {}  # Ánh xạ từ địa chỉ gốc sang chỉ số nút mới
                next_node_index = 0  # Chỉ số tiếp theo cho các nút mới

                # Bước 1: Xử lý các nút và tạo ánh xạ cho địa chỉ sang chỉ số
                for block in cfg_data['blocks']:
                    address = block.get('address', None)
                    if address is not None:
                        assembly = block.get('assembly', [])
                        assembly_code = extract_opcodes(assembly)
                        assembly_list = assembly_code.split('\n')

                        # Mã hóa opcode thành số nguyên
                        encoded_opcodes = label_encoder.transform(assembly_list)

                        # Giới hạn độ dài của opcode hoặc padding nếu cần
                        if len(encoded_opcodes) > max_length:
                            encoded_opcodes = encoded_opcodes[:max_length]  # Cắt bớt nếu quá dài
                        else:
                            # Đệm thêm 0 nếu quá ngắn
                            padding = [0] * (max_length - len(encoded_opcodes))
                            encoded_opcodes = list(encoded_opcodes) + padding
                        # Chuyển đổi đặc trưng opcode thành tensor
                        features_tensor = torch.tensor(encoded_opcodes, dtype=torch.float)

                        node_features.append(features_tensor)

                        # Ánh xạ địa chỉ nút thành chỉ số tuần tự
                        if address not in node_mapping:
                            node_mapping[address] = next_node_index
                            next_node_index += 1

                # Bước 2: Xử lý các cạnh và sử dụng chỉ số được ánh xạ
                edge_list = []
                for edge in cfg_data['edges']:
                    from_node = edge['from']
                    to_node = edge['to']

                    if from_node not in node_mapping or to_node not in node_mapping:
                        # logging.error(f"Cạnh chứa nút không hợp lệ: {from_node} -> {to_node}")
                        continue

                    from_node_idx = node_mapping[from_node]
                    to_node_idx = node_mapping[to_node]

                    edge_list.append((from_node_idx, to_node_idx))

                edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

                if len(edge_index) == 0 or len(node_features) == 0:
                    # logging.warning("Đồ thị không có cạnh hoặc nút hợp lệ, bỏ qua.")
                    continue

            # Chuyển đổi đặc trưng của nút thành tensor
            node_features = torch.stack(node_features)

            graph_data = Data(x=node_features, edge_index=edge_index, y=torch.tensor([graph_label], dtype=torch.float))
            data_list.append(graph_data)

        except Exception as e:
            logging.error(f"Lỗi khi xử lý hàng: {e}")

    return data_list

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)  # Batch Normalization
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)  # Batch Normalization
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(0.5)  # Dropout for regularization


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index).relu()
        x = self.bn2(x)
        x = global_mean_pool(x, data.batch)
        return torch.sigmoid(self.fc(x))

def train_gnn(train_data, val_data, test_data, input_dim):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(input_dim, 64, 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # Reduce LR every 10 epochs
    criterion = torch.nn.BCEWithLogitsLoss()

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)
    test_loader = DataLoader(test_data, batch_size=32)

    best_val_loss = float('inf')
    patience, patience_counter = 5, 0

    for epoch in range(50):
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data).squeeze()
            loss = criterion(output, data.y.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

        # Early stopping logic based on validation loss
        # val_loss = evaluate_with_metrics(model, val_loader, device)
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     patience_counter = 0
        # else:
        #     patience_counter += 1

        # if patience_counter >= patience:
        #     print(f"Early stopping at epoch {epoch + 1}")
        #     break

    # Final evaluation on test set with F1-score and classification report
    print("\nEvaluating on Test Set...")
    evaluate_with_metrics(model, test_loader, device)


def evaluate_with_metrics(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data).squeeze()
            preds = (torch.sigmoid(output) > 0.5).float()  # Binary threshold at 0.5

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

    # Convert predictions and labels to binary for F1-score and report
    all_preds_binary = [int(pred) for pred in all_preds]
    all_labels_binary = [int(label) for label in all_labels]

    # Calculate F1-score and generate classification report
    f1 = f1_score(all_labels_binary, all_preds_binary, average='weighted')
    print(f"Node Test F1 Score: {f1:.5f}")
    print("Node Test Classification Report:")
    print(classification_report(all_labels_binary, all_preds_binary, digits=5))

    # Return F1 score for further tracking if needed
    return f1


def main():
    # Load data
    train_df = pd.read_csv('../data/big-vul_dataset/train_graph.csv')
    val_df = pd.read_csv('../data/big-vul_dataset/val_graph.csv')
    test_df = pd.read_csv('../data/big-vul_dataset/test_graph.csv')

    # Extract all opcodes from the data directly from the JSON content
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

    # Fit the LabelEncoder with the collected opcodes
    label_encoder = LabelEncoder()
    label_encoder.fit(all_opcodes)

    # Find the maximum opcode length
    max_length = find_max_length(train_df, val_df, test_df)

    # Create graph data for each dataset
    train_data = create_graph_data(train_df, label_encoder, max_length)
    # Check label distribution in the train set
    train_labels = [data.y.item() for data in train_data]
    print("Train Set Label Distribution:", pd.Series(train_labels).value_counts())

    val_data = create_graph_data(val_df, label_encoder, max_length)
    # Check label distribution in the val set
    val_labels = [data.y.item() for data in val_data]
    print("Val Set Label Distribution:", pd.Series(val_labels).value_counts())

    test_data = create_graph_data(test_df, label_encoder, max_length)
    # Check label distribution in the test set
    test_labels = [data.y.item() for data in test_data]
    print("Test Set Label Distribution:", pd.Series(test_labels).value_counts())


    print("Test data length:", len(test_data))

    # Train the GNN model
    input_dim = max_length  # Assuming input_dim matches max opcode length
    train_gnn(train_data, val_data, test_data, input_dim)

if __name__ == "__main__":
    main()
