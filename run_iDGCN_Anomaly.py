import utils
import torch
import networkx as nx
from anomaly import ChebGCN
import time
import csv


graph_name = 'libra'
requested_signals=['f_amount_in', 'f_amount_out']
weights_normalization = 'log'
signals_normalization = None

hidden_dimension = 2
num_layers = 1
        
G, _, _, _, signals, labels, weighted_labels = utils.setup_ext(requested_signals=requested_signals, \
                                                                weights_normalization=weights_normalization, \
                                                                signals_normalization=signals_normalization)
signals = torch.tensor(signals, dtype=torch.float32)
size_in = signals.shape[1]
size_out = signals.shape[1]

A = nx.to_scipy_sparse_array(G, format="coo")
row, col, data = A.row, A.col, A.data
L = utils.intensityLaplacian(row, col, G.number_of_nodes(), data, renormalize=True, lambda_max=2.0)
L = utils.cnv_sparse_mat_to_coo_tensor(L)

best_auc = 0
best_lr = ''
best_nodes_01 = 0
best_nodes_02 = 0
best_nodes_05 = 0
best_nodes_1 = 0
best_tpr_01 = 0
best_tpr_02 = 0
best_tpr_05 = 0
best_tpr_1 = 0

utils.write_csv_header('./tmp/res/Anomaly_full_rounds/full_iDGCN.csv')
for lr in [0.001, 0.005, 0.01, 0.05]:
    if lr == 0.001:
        strLr = '0.001'
    if lr == 0.005:
        strLr = '0.005'
    if lr == 0.01:
        strLr = '0.01'
    if lr == 0.05:
        strLr = '0.05'
    nodes_01 = []
    nodes_02 = []
    nodes_05 = []
    nodes_1 = []
    tpr_01 = []
    tpr_02 = []
    tpr_05 = []
    tpr_1 = []
    auc = []
    csvFileName =  './tmp/res/Anomaly_' + graph_name + '/' + str(hidden_dimension) + ' ' + strLr + '_iDGCN.csv'
    utils.write_csv_header(csvFileName)
    for round in range(1, 11):
        csvData = [graph_name, str(nx.number_of_edges(G) / nx.number_of_nodes(G)), signals_normalization if signals_normalization != None else 'No', \
                weights_normalization if weights_normalization != None else 'No', 'Yes', \
                str(1), str(num_layers), str(hidden_dimension), strLr]
        
        model = ChebGCN(size_in=size_in, size_out=size_out, hidden_dim=hidden_dimension, nb_layers=num_layers, K=1, enable_bias=False, droprate=0.5)                              

        # Loss function
        criterion = torch.nn.L1Loss(reduction="none")
        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

        num_epochs = 100
        startTime = time.time()
        losses = []

        best_err = 9223372036854775807 # 2^63 - 1
        early_stopping = 0
        for epoch in range(num_epochs):
            if early_stopping > 10:
                break
            model.train()
            optimizer.zero_grad()
            output = model(signals, L)
            loss = criterion(output, signals)
            loss.sum().backward()
            optimizer.step()
            l = loss.sum().item()
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {l}')
            if l < best_err:
                best_err = l
            else:
                early_stopping += 1
              
        stopTime = time.time()
        csvData += [epoch+1]
        csvData += [str(l)]
        csvData += [str(stopTime - startTime)]

        model.eval()
        with torch.no_grad():
            outEval = model(signals, L)
            err = criterion(outEval, signals)
            print("Libra graph density: ", nx.number_of_edges(G) / nx.number_of_nodes(G))
            data = utils.evaluate(labels, weighted_labels, err)
            csvData += data
            nodes_01.append(float(data[0]))
            nodes_02.append(float(data[3]))
            nodes_05.append(float(data[6]))
            nodes_1.append(float(data[9]))
            tpr_01.append(float(data[2]))
            tpr_02.append(float(data[5]))
            tpr_05.append(float(data[8]))
            tpr_1.append(float(data[11]))
            auc.append(float(data[12]))

        with open(csvFileName, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(csvData)
        with open('./tmp/res/Anomaly_full_rounds/full_iDGCN.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(csvData)

    if best_auc < sum(auc) / 10:
        best_auc = sum(auc) / 10
        best_lr = strLr
        best_nodes_01 = sum(nodes_01) / 10
        best_nodes_02 = sum(nodes_02) / 10
        best_nodes_05 = sum(nodes_05) / 10
        best_nodes_1 = sum(nodes_1) / 10
        best_tpr_01 = sum(tpr_01) / 10
        best_tpr_02 = sum(tpr_02) / 10
        best_tpr_05 = sum(tpr_05) / 10
        best_tpr_1 = sum(tpr_1) / 10


with open('./tmp/res/Anomaly_full_rounds/full_iDGCN.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    csvData = [graph_name, str(nx.number_of_edges(G) / nx.number_of_nodes(G)), signals_normalization if signals_normalization != None else 'No', \
                weights_normalization if weights_normalization != None else 'No', 'Yes', \
                str(1), str(num_layers), str(hidden_dimension), strLr]
    csvData += [' ', ' ', ' ']
    csvData += [str(best_nodes_01), ' ', str(best_tpr_01), str(best_nodes_02), ' ', str(best_tpr_02), str(best_nodes_05), ' ', \
                    str(best_tpr_05), str(best_nodes_1), ' ', str(best_tpr_1),\
                    str(best_auc), ' ']
    writer.writerow(csvData)

    