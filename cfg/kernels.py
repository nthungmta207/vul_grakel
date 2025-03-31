import networkx as nx
from grakel import Graph, GraphKernel
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from joblib import Parallel, delayed
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.gaussian_process.kernels import RBF

def process_in_batches(data, labels, batch_size):
    num_batches = (len(data) + batch_size - 1) // batch_size
    print(f"Processing data in {num_batches} batches")
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size], labels[i:i + batch_size]

def convert_cfg_to_grakel(cfg):
    # Create adjacency matrix
    adjacency_matrix = nx.adjacency_matrix(cfg).todense()

    # Create node labels
    node_labels = {i: cfg.nodes[node]['label'] for i, node in enumerate(cfg.nodes)}

    # Create edge labels
    edge_labels = {}
    for i, (from_node, to_node, data) in enumerate(cfg.edges(data=True)):
        edge_index = (list(cfg.nodes).index(from_node), list(cfg.nodes).index(to_node))
        edge_labels[edge_index] = data.get('label', '')

    return Graph(adjacency_matrix, node_labels=node_labels, edge_labels=edge_labels)

def wl_sp(train_cfg,train_graph_labels, batch_size, test_graph_labels, test_cfg):
    print("Using Weisfeiler-Lehman and Shortest Path kernels")
    # Define multiple kernels
    gk1 = GraphKernel(kernel={"name": "weisfeiler_lehman", "n_iter": 5}, normalize=True, n_jobs=-1)
    gk2 = GraphKernel(kernel={"name": "shortest_path"}, normalize=True, n_jobs=-1)

    svc = SVC(kernel="precomputed", probability=True, random_state=42)
    clf_graph = GridSearchCV(svc, param_grid={'C': [0.1, 1, 10, 100, 1000]}, cv=5, n_jobs=-1)

    for i, (train_batch, train_labels_batch) in enumerate(
            process_in_batches(train_cfg, train_graph_labels, batch_size)):
        train_graphs = Parallel(n_jobs=-1)(delayed(convert_cfg_to_grakel)(cfg) for cfg in train_batch)

        # Compute kernel matrices for both kernels
        K_train_batch_1 = gk1.fit_transform(train_graphs)
        K_train_batch_2 = gk2.fit_transform(train_graphs)

        # Combine the kernel matrices (you can adjust the combination method)
        K_train_batch = (K_train_batch_1 + K_train_batch_2) / 2

        train_labels_batch = np.array(train_labels_batch).flatten()

        print(
            f"Batch {i} - K_train_batch shape: {K_train_batch.shape}, train_labels_batch shape: {train_labels_batch.shape}")

        clf_graph.fit(K_train_batch, train_labels_batch)

        with open(f'./saved_kernel/K_train_batch_{i}.pkl', 'wb') as f:
            pickle.dump(K_train_batch, f)
        with open(f'./saved_kernel/svm_model_graph_batch_{i}.pkl', 'wb') as f:
            pickle.dump(clf_graph, f)

    # val_graphs = Parallel(n_jobs=-1)(delayed(convert_cfg_to_grakel)(cfg) for cfg in val_cfg)
    # K_val_1 = gk1.transform(val_graphs)
    # K_val_2 = gk2.transform(val_graphs)
    # K_val = (K_val_1 + K_val_2) / 2
    # val_pred = clf_graph.predict(K_val)
    # print("Graph Validation Accuracy:", accuracy_score(val_graph_labels, val_pred))
    # print("Graph Validation Classification Report:")
    # print(classification_report(val_graph_labels, val_pred, digits=5))

    test_graphs = Parallel(n_jobs=-1)(delayed(convert_cfg_to_grakel)(cfg) for cfg in test_cfg)
    K_test_1 = gk1.transform(test_graphs)
    K_test_2 = gk2.transform(test_graphs)
    K_test = (K_test_1 + K_test_2) / 2
    test_pred = clf_graph.predict(K_test)

    print("Graph Test Accuracy:", accuracy_score(test_graph_labels, test_pred))
    f1 = f1_score(test_graph_labels, test_pred, average='weighted')  # Hoặc 'macro', 'micro', tùy theo yêu cầu
    precision = precision_score(test_graph_labels, test_pred, average='weighted')
    recall = recall_score(test_graph_labels, test_pred, average='weighted')
    print(f"F1 Score: {f1:.5f}")
    print(f"Precision: {precision:.5f}")
    print(f"Recall: {recall:.5f}")
    print("Graph Test Classification Report:")
    print(classification_report(test_graph_labels, test_pred, digits=5))
    # test_pred = clf_graph.predict_proba(K_test)
    # print("Graph Test Classification Report:", test_pred)

    with open('./saved_kernel/K_train.pkl', 'wb') as f:
        pickle.dump(K_train_batch, f)
    with open('./saved_kernel/K_test.pkl', 'wb') as f:
        pickle.dump(K_test, f)

    # Extract correctly predicted CFGs
    correct_test_indices = [i for i, (true, pred) in enumerate(zip(test_graph_labels, test_pred)) if
                            true == int(pred) and pred == 1]

    correct_test_cfgs = [test_cfg[i] for i in correct_test_indices]
    correct_test_labels = [test_graph_labels[i] for i in correct_test_indices]

    print(f"Number of correctly predicted CFGs in test set: {len(correct_test_cfgs)}")
    return correct_test_indices

def wl(train_cfg,train_graph_labels, batch_size, test_graph_labels, test_cfg):
    # Define Weisfeiler-Lehman kernel
    print("Using Weisfeiler-Lehman kernel")
    gk = GraphKernel(kernel={"name": "weisfeiler_lehman", "n_iter": 5}, normalize=True, n_jobs=-1)

    svc = SVC(kernel="precomputed", probability=True, random_state=42)
    clf_graph = GridSearchCV(svc, param_grid={'C': [0.1, 1, 10, 100, 1000]}, cv=5, n_jobs=-1)

    for i, (train_batch, train_labels_batch) in enumerate(
            process_in_batches(train_cfg, train_graph_labels, batch_size)):
        train_graphs = Parallel(n_jobs=-1)(delayed(convert_cfg_to_grakel)(cfg) for cfg in train_batch)

        # Compute the kernel matrix using the Weisfeiler-Lehman kernel
        K_train_batch = gk.fit_transform(train_graphs)

        train_labels_batch = np.array(train_labels_batch).flatten()

        print(
            f"Batch {i} - K_train_batch shape: {K_train_batch.shape}, train_labels_batch shape: {train_labels_batch.shape}")

        clf_graph.fit(K_train_batch, train_labels_batch)

        with open(f'./saved_kernel/K_train_batch_{i}.pkl', 'wb') as f:
            pickle.dump(K_train_batch, f)
        with open(f'./saved_kernel/svm_model_graph_batch_{i}.pkl', 'wb') as f:
            pickle.dump(clf_graph, f)

    test_graphs = Parallel(n_jobs=-1)(delayed(convert_cfg_to_grakel)(cfg) for cfg in test_cfg)
    K_test = gk.transform(test_graphs)
    test_pred = clf_graph.predict(K_test)

    print("Graph Test Accuracy:", accuracy_score(test_graph_labels, test_pred))
    f1 = f1_score(test_graph_labels, test_pred, average='weighted')  # Hoặc 'macro', 'micro', tùy theo yêu cầu
    print(f"F1 Score: {f1:.5f}")
    print("Graph Test Classification Report:")
    print(classification_report(test_graph_labels, test_pred, digits=5))

    with open('./saved_kernel/K_train.pkl', 'wb') as f:
        pickle.dump(K_train_batch, f)
    with open('./saved_kernel/K_test.pkl', 'wb') as f:
        pickle.dump(K_test, f)

    # Extract correctly predicted CFGs
    correct_test_indices = [i for i, (true, pred) in enumerate(zip(test_graph_labels, test_pred)) if
                            true == int(pred) and pred == 1]

    correct_test_cfgs = [test_cfg[i] for i in correct_test_indices]
    correct_test_labels = [test_graph_labels[i] for i in correct_test_indices]

    print(f"Number of correctly predicted CFGs in test set: {len(correct_test_cfgs)}")
    return correct_test_indices

def subgraph(train_cfg,train_graph_labels, batch_size, test_graph_labels, test_cfg):
    # Define Subgraph Matching kernel
    print("Using Subgraph Matching kernel")
    gk = GraphKernel(kernel={"name": "subgraph_matching"}, normalize=True, n_jobs=-1)

    svc = SVC(kernel="precomputed", probability=True, random_state=42)
    clf_graph = GridSearchCV(svc, param_grid={'C': [0.1, 1, 10, 100, 1000]}, cv=5, n_jobs=-1)

    for i, (train_batch, train_labels_batch) in enumerate(
            process_in_batches(train_cfg, train_graph_labels, batch_size)):
        train_graphs = Parallel(n_jobs=-1)(delayed(convert_cfg_to_grakel)(cfg) for cfg in train_batch)

        # Compute the kernel matrix using the Weisfeiler-Lehman kernel
        K_train_batch = gk.fit_transform(train_graphs)

        train_labels_batch = np.array(train_labels_batch).flatten()

        print(
            f"Batch {i} - K_train_batch shape: {K_train_batch.shape}, train_labels_batch shape: {train_labels_batch.shape}")

        clf_graph.fit(K_train_batch, train_labels_batch)

        with open(f'./saved_kernel/K_train_batch_{i}.pkl', 'wb') as f:
            pickle.dump(K_train_batch, f)
        with open(f'./saved_kernel/svm_model_graph_batch_{i}.pkl', 'wb') as f:
            pickle.dump(clf_graph, f)

    test_graphs = Parallel(n_jobs=-1)(delayed(convert_cfg_to_grakel)(cfg) for cfg in test_cfg)
    K_test = gk.transform(test_graphs)
    test_pred = clf_graph.predict(K_test)

    print("Graph Test Accuracy:", accuracy_score(test_graph_labels, test_pred))
    f1 = f1_score(test_graph_labels, test_pred, average='weighted')  # Hoặc 'macro', 'micro', tùy theo yêu cầu
    print(f"F1 Score: {f1:.5f}")
    print("Graph Test Classification Report:")
    print(classification_report(test_graph_labels, test_pred, digits=5))

    with open('./saved_kernel/K_train.pkl', 'wb') as f:
        pickle.dump(K_train_batch, f)
    with open('./saved_kernel/K_test.pkl', 'wb') as f:
        pickle.dump(K_test, f)

    # Extract correctly predicted CFGs
    correct_test_indices = [i for i, (true, pred) in enumerate(zip(test_graph_labels, test_pred)) if
                            true == int(pred) and pred == 1]

    correct_test_cfgs = [test_cfg[i] for i in correct_test_indices]
    correct_test_labels = [test_graph_labels[i] for i in correct_test_indices]

    print(f"Number of correctly predicted CFGs in test set: {len(correct_test_cfgs)}")
    return correct_test_indices


def wl_oa_rbf(train_cfg, train_graph_labels, batch_size, test_graph_labels, test_cfg):
    # Define Weisfeiler-Lehman kernel
    print("Using Weisfeiler-Lehman Optimal Assignment kernel and RBF kernel")
    gk = GraphKernel(kernel={"name": "weisfeiler_lehman_optimal_assignment", "n_iter": 5}, normalize=True, n_jobs=-1)
    rbf_kernel = RBF()

    scaler = StandardScaler()

    svc = SVC(kernel="precomputed", probability=True, random_state=42)
    clf_graph = GridSearchCV(svc, param_grid={'C': [0.1, 1, 10, 100, 1000]}, cv=5, n_jobs=-1)

    for i, (train_batch, train_labels_batch) in enumerate(
            process_in_batches(train_cfg, train_graph_labels, batch_size)):
        train_graphs = Parallel(n_jobs=-1)(delayed(convert_cfg_to_grakel)(cfg) for cfg in train_batch)

        # Compute the kernel matrix using the Weisfeiler-Lehman kernel
        K_train_batch = gk.fit_transform(train_graphs)

        # Apply RBF kernel on the Weisfeiler-Lehman kernel matrix
        K_train_batch_rbf = rbf_kernel(K_train_batch)


        train_labels_batch = np.array(train_labels_batch).flatten()

        print(f"Batch {i} - K_train_batch shape: {K_train_batch.shape}, train_labels_batch shape: {train_labels_batch.shape}")

        clf_graph.fit(K_train_batch_rbf, train_labels_batch)

        with open(f'./saved_kernel/K_train_batch_{i}.pkl', 'wb') as f:
            pickle.dump(K_train_batch_rbf, f)
        with open(f'./saved_kernel/svm_model_graph_batch_{i}.pkl', 'wb') as f:
            pickle.dump(clf_graph, f)

    test_graphs = Parallel(n_jobs=-1)(delayed(convert_cfg_to_grakel)(cfg) for cfg in test_cfg)
    K_test = gk.transform(test_graphs)

    # Apply RBF kernel on the Weisfeiler-Lehman kernel matrix
    K_test_rbf = rbf_kernel(K_test, K_train_batch)


    test_pred = clf_graph.predict(K_test_rbf)

    print("Graph Test Accuracy:", accuracy_score(test_graph_labels, test_pred))
    f1 = f1_score(test_graph_labels, test_pred, average='weighted')  # Hoặc 'macro', 'micro', tùy theo yêu cầu
    print(f"F1 Score: {f1:.5f}")
    print("Graph Test Classification Report:")
    print(classification_report(test_graph_labels, test_pred, digits=5))

    with open('./saved_kernel/K_train.pkl', 'wb') as f:
        pickle.dump(K_train_batch_rbf, f)
    with open('./saved_kernel/K_test.pkl', 'wb') as f:
        pickle.dump(K_test_rbf, f)

    # Extract correctly predicted CFGs
    correct_test_indices = [i for i, (true, pred) in enumerate(zip(test_graph_labels, test_pred)) if
                            true == int(pred) and pred == 1]

    correct_test_cfgs = [test_cfg[i] for i in correct_test_indices]
    correct_test_labels = [test_graph_labels[i] for i in correct_test_indices]

    print(f"Number of correctly predicted CFGs in test set: {len(correct_test_cfgs)}")
    return correct_test_indices
def wl_rbf(train_cfg, train_graph_labels, batch_size, test_graph_labels, test_cfg):
    # Define Weisfeiler-Lehman kernel
    print("Using Weisfeiler-Lehman kernel and RBF kernel")
    gk = GraphKernel(kernel={"name": "weisfeiler_lehman", "n_iter": 5}, normalize=True, n_jobs=-1)
    rbf_kernel = RBF()

    scaler = StandardScaler()

    svc = SVC(kernel="precomputed", probability=True, random_state=42)
    clf_graph = GridSearchCV(svc, param_grid={'C': [0.1, 1, 10, 100, 1000]}, cv=5, n_jobs=-1)

    for i, (train_batch, train_labels_batch) in enumerate(
            process_in_batches(train_cfg, train_graph_labels, batch_size)):
        train_graphs = Parallel(n_jobs=-1)(delayed(convert_cfg_to_grakel)(cfg) for cfg in train_batch)

        # Compute the kernel matrix using the Weisfeiler-Lehman kernel
        K_train_batch = gk.fit_transform(train_graphs)

        # Apply RBF kernel on the Weisfeiler-Lehman kernel matrix
        K_train_batch_rbf = rbf_kernel(K_train_batch)


        train_labels_batch = np.array(train_labels_batch).flatten()

        print(f"Batch {i} - K_train_batch shape: {K_train_batch.shape}, train_labels_batch shape: {train_labels_batch.shape}")

        clf_graph.fit(K_train_batch_rbf, train_labels_batch)

        with open(f'./saved_kernel/K_train_batch_{i}.pkl', 'wb') as f:
            pickle.dump(K_train_batch_rbf, f)
        with open(f'./saved_kernel/svm_model_graph_batch_{i}.pkl', 'wb') as f:
            pickle.dump(clf_graph, f)

    test_graphs = Parallel(n_jobs=-1)(delayed(convert_cfg_to_grakel)(cfg) for cfg in test_cfg)
    K_test = gk.transform(test_graphs)

    # Apply RBF kernel on the Weisfeiler-Lehman kernel matrix
    K_test_rbf = rbf_kernel(K_test, K_train_batch)


    test_pred = clf_graph.predict(K_test_rbf)

    print("Graph Test Accuracy:", accuracy_score(test_graph_labels, test_pred))
    f1 = f1_score(test_graph_labels, test_pred, average='weighted')  # Hoặc 'macro', 'micro', tùy theo yêu cầu
    print(f"F1 Score: {f1:.5f}")
    print("Graph Test Classification Report:")
    print(classification_report(test_graph_labels, test_pred, digits=5))

    with open('./saved_kernel/K_train.pkl', 'wb') as f:
        pickle.dump(K_train_batch_rbf, f)
    with open('./saved_kernel/K_test.pkl', 'wb') as f:
        pickle.dump(K_test_rbf, f)

    # Extract correctly predicted CFGs
    correct_test_indices = [i for i, (true, pred) in enumerate(zip(test_graph_labels, test_pred)) if
                            true == int(pred) and pred == 1]

    correct_test_cfgs = [test_cfg[i] for i in correct_test_indices]
    correct_test_labels = [test_graph_labels[i] for i in correct_test_indices]

    print(f"Number of correctly predicted CFGs in test set: {len(correct_test_cfgs)}")
    return correct_test_indices

def wl_rbf_sp(train_cfg, train_graph_labels, batch_size, test_graph_labels, test_cfg):
    # Define Weisfeiler-Lehman, RBF and Shortest Path kernels
    print("Using Weisfeiler-Lehman kernel, RBF kernel, and Shortest Path kernel")
    gk_wl = GraphKernel(kernel={"name": "weisfeiler_lehman", "n_iter": 5}, normalize=False, n_jobs=-1)
    rbf_kernel = RBF()
    gk_sp = GraphKernel(kernel={"name": "shortest_path"}, normalize=False, n_jobs=-1)

    svc = SVC(kernel="precomputed", probability=True, random_state=42)
    clf_graph = GridSearchCV(svc, param_grid={'C': [0.1, 1, 10, 100, 1000]}, cv=5, n_jobs=-1)

    for i, (train_batch, train_labels_batch) in enumerate(
            process_in_batches(train_cfg, train_graph_labels, batch_size)):
        train_graphs = Parallel(n_jobs=-1)(delayed(convert_cfg_to_grakel)(cfg) for cfg in train_batch)

        # Compute the kernel matrices using the Weisfeiler-Lehman and Shortest Path kernels
        K_train_batch_wl = gk_wl.fit_transform(train_graphs)
        K_train_batch_sp = gk_sp.fit_transform(train_graphs)

        # Combine the kernel matrices (you can adjust the combination method)
        K_train_batch_combined = (K_train_batch_wl + K_train_batch_sp) / 2

        # Apply RBF kernel on the combined kernel matrix
        K_train_batch_combined_rbf = rbf_kernel(K_train_batch_combined)

        train_labels_batch = np.array(train_labels_batch).flatten()

        print(f"Batch {i} - K_train_batch_combined shape: {K_train_batch_combined.shape}, train_labels_batch shape: {train_labels_batch.shape}")

        clf_graph.fit(K_train_batch_combined_rbf, train_labels_batch)

        with open(f'./saved_kernel/K_train_batch_{i}.pkl', 'wb') as f:
            pickle.dump(K_train_batch_combined_rbf, f)
        with open(f'./saved_kernel/svm_model_graph_batch_{i}.pkl', 'wb') as f:
            pickle.dump(clf_graph, f)

    test_graphs = Parallel(n_jobs=-1)(delayed(convert_cfg_to_grakel)(cfg) for cfg in test_cfg)
    K_test_wl = gk_wl.transform(test_graphs)
    K_test_sp = gk_sp.transform(test_graphs)

    # Combine the test kernel matrices
    K_test_combined = (K_test_wl + K_test_sp) / 2

    # Apply RBF kernel on the combined test kernel matrix
    K_test_combined_rbf = rbf_kernel(K_test_combined, K_train_batch_combined)

    test_pred = clf_graph.predict(K_test_combined_rbf)

    print("Graph Test Accuracy:", accuracy_score(test_graph_labels, test_pred))
    f1 = f1_score(test_graph_labels, test_pred, average='weighted')
    print(f"F1 Score: {f1:.5f}")
    print("Graph Test Classification Report:")
    print(classification_report(test_graph_labels, test_pred, digits=5))

    with open('./saved_kernel/K_train.pkl', 'wb') as f:
        pickle.dump(K_train_batch_combined_rbf, f)
    with open('./saved_kernel/K_test.pkl', 'wb') as f:
        pickle.dump(K_test_combined_rbf, f)

    # Extract correctly predicted CFGs
    correct_test_indices = [i for i, (true, pred) in enumerate(zip(test_graph_labels, test_pred)) if
                            true == int(pred) and pred == 1]

    correct_test_cfgs = [test_cfg[i] for i in correct_test_indices]
    correct_test_labels = [test_graph_labels[i] for i in correct_test_indices]

    print(f"Number of correctly predicted CFGs in test set: {len(correct_test_cfgs)}")
    return correct_test_indices
def shortest_path_kernel(train_cfg, train_graph_labels, batch_size, test_graph_labels, test_cfg):
    # Define Shortest-Path kernel
    print("Using Shortest-Path kernel")
    gk = GraphKernel(kernel={"name": "shortest_path"}, normalize=True, n_jobs=-1)

    svc = SVC(kernel="precomputed", probability=True, random_state=42)
    clf_graph = GridSearchCV(svc, param_grid={'C': [0.1, 1, 10, 100, 1000]}, cv=5, n_jobs=-1)

    for i, (train_batch, train_labels_batch) in enumerate(
            process_in_batches(train_cfg, train_graph_labels, batch_size)):
        train_graphs = Parallel(n_jobs=-1)(delayed(convert_cfg_to_grakel)(cfg) for cfg in train_batch)

        # Compute the kernel matrix using the Shortest-Path kernel
        K_train_batch = gk.fit_transform(train_graphs)

        train_labels_batch = np.array(train_labels_batch).flatten()

        print(
            f"Batch {i} - K_train_batch shape: {K_train_batch.shape}, train_labels_batch shape: {train_labels_batch.shape}")

        clf_graph.fit(K_train_batch, train_labels_batch)

        with open(f'./saved_kernel/K_train_batch_{i}.pkl', 'wb') as f:
            pickle.dump(K_train_batch, f)
        with open(f'./saved_kernel/svm_model_graph_batch_{i}.pkl', 'wb') as f:
            pickle.dump(clf_graph, f)

    test_graphs = Parallel(n_jobs=-1)(delayed(convert_cfg_to_grakel)(cfg) for cfg in test_cfg)
    K_test = gk.transform(test_graphs)
    test_pred = clf_graph.predict(K_test)

    print("Graph Test Accuracy:", accuracy_score(test_graph_labels, test_pred))
    f1 = f1_score(test_graph_labels, test_pred, average='weighted')  # Hoặc 'macro', 'micro', tùy theo yêu cầu
    precision = precision_score(test_graph_labels, test_pred, average='weighted')
    recall = recall_score(test_graph_labels, test_pred, average='weighted')
    print(f"F1 Score: {f1:.5f}")
    print(f"Precision: {precision:.5f}")
    print(f"Recall: {recall:.5f}")
    print("Graph Test Classification Report:")
    print(classification_report(test_graph_labels, test_pred, digits=5))

    with open('./saved_kernel/K_train.pkl', 'wb') as f:
        pickle.dump(K_train_batch, f)
    with open('./saved_kernel/K_test.pkl', 'wb') as f:
        pickle.dump(K_test, f)

    # Extract correctly predicted CFGs
    correct_test_indices = [i for i, (true, pred) in enumerate(zip(test_graph_labels, test_pred)) if
                            true == int(pred) and pred == 1]

    correct_test_cfgs = [test_cfg[i] for i in correct_test_indices]
    correct_test_labels = [test_graph_labels[i] for i in correct_test_indices]

    print(f"Number of correctly predicted CFGs in test set: {len(correct_test_cfgs)}")
    return correct_test_indices

def wl_oa(train_cfg, train_graph_labels, batch_size, test_graph_labels, test_cfg):
    # Define Weisfeiler-Lehman kernel
    print("Using Weisfeiler-Lehman Optimal Assignment kernel")
    gk = GraphKernel(kernel={"name": "weisfeiler_lehman_optimal_assignment", "n_iter": 5}, normalize=True, n_jobs=-1)

    svc = SVC(kernel="precomputed", probability=True, random_state=42)
    clf_graph = GridSearchCV(svc, param_grid={'C': [0.1, 1, 10, 100, 1000]}, cv=5, n_jobs=-1)

    for i, (train_batch, train_labels_batch) in enumerate(
            process_in_batches(train_cfg, train_graph_labels, batch_size)):
        train_graphs = Parallel(n_jobs=-1)(delayed(convert_cfg_to_grakel)(cfg) for cfg in train_batch)

        # Compute the kernel matrix using the Weisfeiler-Lehman kernel
        K_train_batch = gk.fit_transform(train_graphs)

        train_labels_batch = np.array(train_labels_batch).flatten()

        print(
            f"Batch {i} - K_train_batch shape: {K_train_batch.shape}, train_labels_batch shape: {train_labels_batch.shape}")

        clf_graph.fit(K_train_batch, train_labels_batch)

        with open(f'./saved_kernel/K_train_batch_{i}.pkl', 'wb') as f:
            pickle.dump(K_train_batch, f)
        with open(f'./saved_kernel/svm_model_graph_batch_{i}.pkl', 'wb') as f:
            pickle.dump(clf_graph, f)

    test_graphs = Parallel(n_jobs=-1)(delayed(convert_cfg_to_grakel)(cfg) for cfg in test_cfg)
    K_test = gk.transform(test_graphs)
    test_pred = clf_graph.predict(K_test)

    print("Graph Test Accuracy:", accuracy_score(test_graph_labels, test_pred))
    f1 = f1_score(test_graph_labels, test_pred, average='weighted')  # Hoặc 'macro', 'micro', tùy theo yêu cầu
    precision = precision_score(test_graph_labels, test_pred, average='weighted')
    recall = recall_score(test_graph_labels, test_pred, average='weighted')
    print(f"F1 Score: {f1:.5f}")
    print(f"Precision: {precision:.5f}")
    print(f"Recall: {recall:.5f}")
    print("Graph Test Classification Report:")
    print(classification_report(test_graph_labels, test_pred, digits=5))

    with open('./saved_kernel/K_train.pkl', 'wb') as f:
        pickle.dump(K_train_batch, f)
    with open('./saved_kernel/K_test.pkl', 'wb') as f:
        pickle.dump(K_test, f)

    # Extract correctly predicted CFGs
    correct_test_indices = [i for i, (true, pred) in enumerate(zip(test_graph_labels, test_pred)) if
                            true == int(pred) and pred == 1]

    correct_test_cfgs = [test_cfg[i] for i in correct_test_indices]
    correct_test_labels = [test_graph_labels[i] for i in correct_test_indices]

    print(f"Number of correctly predicted CFGs in test set: {len(correct_test_cfgs)}")
    return correct_test_indices


def wl_sp_rbf(train_cfg, train_graph_labels, batch_size, test_graph_labels, test_cfg):
    # Define multiple kernels
    gk1 = GraphKernel(kernel={"name": "weisfeiler_lehman", "n_iter": 5}, normalize=True)
    gk2 = GraphKernel(kernel={"name": "shortest_path"}, normalize=True, n_jobs=-1)
    rbf_kernel = RBF()

    svc = SVC(kernel="precomputed", probability=True, random_state=42)
    clf_graph = GridSearchCV(svc, param_grid={'C': [0.1, 1, 10, 100, 1000]}, cv=5, n_jobs=-1)

    scaler = StandardScaler()

    for i, (train_batch, train_labels_batch) in enumerate(
            process_in_batches(train_cfg, train_graph_labels, batch_size)):
        train_graphs = Parallel(n_jobs=-1)(delayed(convert_cfg_to_grakel)(cfg) for cfg in train_batch)

        # Compute kernel matrices for both kernels
        K_train_batch_1 = gk1.fit_transform(train_graphs)
        K_train_batch_2 = gk2.fit_transform(train_graphs)

        # Combine the graph kernels
        K_train_batch_graph = (K_train_batch_1 + K_train_batch_2) / 2

        # Compute the Gaussian RBF kernel
        K_train_batch_rbf = rbf_kernel(K_train_batch_graph, K_train_batch_graph)

        train_labels_batch = np.array(train_labels_batch).flatten()

        print(f"Batch {i} - K_train_batch shape: {K_train_batch_graph.shape}, train_labels_batch shape: {train_labels_batch.shape}")

        # Feature Scaling
        K_train_batch_scaled = scaler.fit_transform(K_train_batch_rbf)

        clf_graph.fit(K_train_batch_scaled, train_labels_batch)

        with open(f'./saved_kernel/K_train_batch_{i}.pkl', 'wb') as f:
            pickle.dump(K_train_batch_rbf, f)
        with open(f'./saved_kernel/svm_model_graph_batch_{i}.pkl', 'wb') as f:
            pickle.dump(clf_graph, f)

    test_graphs = Parallel(n_jobs=-1)(delayed(convert_cfg_to_grakel)(cfg) for cfg in test_cfg)
    K_test_1 = gk1.transform(test_graphs)
    K_test_2 = gk2.transform(test_graphs)
    K_test_graph = (K_test_1 + K_test_2) / 2

    # Compute the Gaussian RBF kernel for test data
    K_test_rbf = rbf_kernel(K_test_graph, K_train_batch_graph)


    test_pred = clf_graph.predict(K_test_rbf)

    print("Graph Test Accuracy:", accuracy_score(test_graph_labels, test_pred))
    f1 = f1_score(test_graph_labels, test_pred, average='weighted')
    print(f"F1 Score: {f1:.5f}")
    print("Graph Test Classification Report:")
    print(classification_report(test_graph_labels, test_pred, digits=5))

    with open('./saved_kernel/K_train.pkl', 'wb') as f:
        pickle.dump(K_train_batch_rbf, f)
    with open('./saved_kernel/K_test.pkl', 'wb') as f:
        pickle.dump(K_test_rbf, f)

    # Extract correctly predicted CFGs
    correct_test_indices = [i for i, (true, pred) in enumerate(zip(test_graph_labels, test_pred)) if true == int(pred) and pred == 1]

    correct_test_cfgs = [test_cfg[i] for i in correct_test_indices]
    correct_test_labels = [test_graph_labels[i] for i in correct_test_indices]

    print(f"Number of correctly predicted CFGs in test set: {len(correct_test_cfgs)}")
    return correct_test_indices


def wl_oa_sp(train_cfg, train_graph_labels, batch_size, test_graph_labels, test_cfg):
    # Define Weisfeiler-Lehman Optimal Assignment and Shortest Path kernels
    print("Using Weisfeiler-Lehman Optimal Assignment and Shortest Path kernels")
    gk_wl_oa = GraphKernel(kernel=[{"name": "weisfeiler_lehman_optimal_assignment", "n_iter": 5},{"name": "vertex_histogram"}], normalize=True, n_jobs=-1)
    gk_sp = GraphKernel(kernel={"name": "shortest_path"}, normalize=True, n_jobs=-1)

    svc = SVC(kernel="precomputed", probability=True, random_state=42)
    clf_graph = GridSearchCV(svc, param_grid={'C': [0.1, 1, 10, 100, 1000]}, cv=5, n_jobs=-1)

    for i, (train_batch, train_labels_batch) in enumerate(
            process_in_batches(train_cfg, train_graph_labels, batch_size)):
        train_graphs = Parallel(n_jobs=-1)(delayed(convert_cfg_to_grakel)(cfg) for cfg in train_batch)

        # Compute the kernel matrices using Weisfeiler-Lehman Optimal Assignment and Shortest Path kernels
        K_train_batch_wl_oa = gk_wl_oa.fit_transform(train_graphs)
        K_train_batch_sp = gk_sp.fit_transform(train_graphs)

        # Combine the kernel matrices (you can adjust the combination method)
        K_train_batch = (K_train_batch_wl_oa +  2*K_train_batch_sp ) / 3

        train_labels_batch = np.array(train_labels_batch).flatten()

        print(
            f"Batch {i} - K_train_batch shape: {K_train_batch.shape}, train_labels_batch shape: {train_labels_batch.shape}")

        clf_graph.fit(K_train_batch, train_labels_batch)

        with open(f'./saved_kernel/K_train_batch_{i}.pkl', 'wb') as f:
            pickle.dump(K_train_batch, f)
        with open(f'./saved_kernel/svm_model_graph_batch_{i}.pkl', 'wb') as f:
            pickle.dump(clf_graph, f)

    test_graphs = Parallel(n_jobs=-1)(delayed(convert_cfg_to_grakel)(cfg) for cfg in test_cfg)
    K_test_wl_oa = gk_wl_oa.transform(test_graphs)
    K_test_sp = gk_sp.transform(test_graphs)
    K_test = (K_test_wl_oa + K_test_sp) / 2

    test_pred = clf_graph.predict(K_test)

    print("Graph Test Accuracy:", accuracy_score(test_graph_labels, test_pred))
    f1 = f1_score(test_graph_labels, test_pred, average='weighted')  # Hoặc 'macro', 'micro', tùy theo yêu cầu
    precision = precision_score(test_graph_labels, test_pred, average='weighted')
    recall = recall_score(test_graph_labels, test_pred, average='weighted')
    print(f"F1 Score: {f1:.5f}")
    print(f"Precision: {precision:.5f}")
    print(f"Recall: {recall:.5f}")
    print("Graph Test Classification Report:")
    print(classification_report(test_graph_labels, test_pred, digits=5))

    with open('./saved_kernel/K_train.pkl', 'wb') as f:
        pickle.dump(K_train_batch, f)
    with open('./saved_kernel/K_test.pkl', 'wb') as f:
        pickle.dump(K_test, f)

    # Extract correctly predicted CFGs
    correct_test_indices = [i for i, (true, pred) in enumerate(zip(test_graph_labels, test_pred)) if
                            true == int(pred) and pred == 1]

    correct_test_cfgs = [test_cfg[i] for i in correct_test_indices]
    correct_test_labels = [test_graph_labels[i] for i in correct_test_indices]

    print(f"Number of correctly predicted CFGs in test set: {len(correct_test_cfgs)}")
    return correct_test_indices
