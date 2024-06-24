import numpy as np
import random
import torch
import time
from torch_geometric_signed_directed.data import load_directed_real_data

import utils
from sigmanet import SigMaNet_link_prediction_one_laplacian

# select cuda device if available
cuda_device = 0
device = torch.device("cuda:%d" % cuda_device if torch.cuda.is_available() else "cpu")

epochs = 3000
dropout = 0.5
dataset_name = 'telegram'

nb_epochs = [0 for _ in range(360)]
it_epochs = -1
for task in ['existence']: # 'direction']:
    num_class_link = 2
    for lr in [0.001, 0.005, 0.01, 0.05]:
        for num_filter in [16, 32, 64]:
            for layer in [2, 4, 8]:
                current_params = 'lr_' + str(lr) + '_num_filter_' + str(num_filter) + '_layer_' + str(layer)
                print(current_params)

                log_path = './tmp/res/Edge_SigMaNet_' + dataset_name
                random.seed(0)
                torch.manual_seed(0)
                np.random.seed(0)

                if dataset_name in ['telegram']:
                    data = load_directed_real_data(dataset=dataset_name, name=dataset_name).to(device)
                    data = data.to(device)
                elif dataset_name in ['bitcoin_alpha', 'bitcoin_otc']:
                    data = utils.load_signed_real_data_no_negative(dataset=dataset_name, root='./tmp/').to(device)
                else:
                    raise Exception("Wrong dataset.")

                edge_index = data.edge_index
                size = torch.max(edge_index).item() + 1
                data.num_nodes = size

                datasets = utils.link_class_split_new(data, prob_val=0.05, prob_test=0.15, splits=10, task=task)

                avg_acc_overall = [0.0 for _ in range(10)]
                avg_err_overall = [100000000000.0 for _ in range(10)]
                validation_error_model_acc = [0.0 for _ in range(10)]
                validation_acc_model_err = [100000000000.0 for _ in range(10)]
                models = []
                for i in range(10):
                    it_epochs += 1
                    log_str_full = ''
                    ########################################
                    # get hermitian laplacian
                    ########################################
                    edges = datasets[i]['graph']
                    weights = datasets[i]['weights']
                    
                    X_real = utils.in_out_degree(edges, size, weights).to(device)
                    X_img = X_real.clone()

                    edge_index, norm_real, norm_imag = utils.process_magnetic_laplacian(edge_index=edges, gcn=True, net_flow=True,\
                                                            x_real=X_real, edge_weight=weights, normalization='sym', return_lambda_max=False)
                    ########################################
                    # initialize model and load dataset
                    ########################################
                    model = SigMaNet_link_prediction_one_laplacian(K=1, num_features=2, hidden=num_filter, label_dim=num_class_link,\
                            i_complex = False,  layer=layer, follow_math=True, gcn=True, net_flow=True, unwind=True, edge_index=edge_index,\
                            norm_real=norm_real, norm_imag=norm_imag).to(device)    
                    model = model.to(device)

                    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

                    y_train = datasets[i]['train']['label']
                    y_val   = datasets[i]['val']['label']
                    y_test  = datasets[i]['test']['label']
                    y_train = y_train.long().to(device)
                    y_val   = y_val.long().to(device)
                    y_test  = y_test.long().to(device)

                    train_index = datasets[i]['train']['edges'].to(device)
                    val_index = datasets[i]['val']['edges'].to(device)
                    test_index = datasets[i]['test']['edges'].to(device)

                    #################################
                    # Train/Validation/Test
                    #################################
                    best_test_err = 100000000000.0
                    best_test_acc = 0.0
                    early_stopping = 0
                    for epoch in range(epochs):
                        start_time = time.time()
                        if early_stopping > 500:
                            break
                        nb_epochs[it_epochs] = epoch
                        ####################
                        # Train
                        ####################
                        train_loss, train_acc = 0.0, 0.0
                        model.train()
                        out = model(X_real, X_img, train_index)
                        train_loss = torch.nn.functional.nll_loss(out, y_train)
                        pred_label = out.max(dim = 1)[1]            
                        train_acc  = utils.acc(pred_label, y_train)
                        opt.zero_grad()
                        train_loss.backward()
                        opt.step()

                        ####################
                        # Validation
                        ####################
                        val_loss, val_acc = 0.0, 0.0
                        model.eval()
                        out = model(X_real, X_img, val_index)
                        val_loss = torch.nn.functional.nll_loss(out, y_val)
                        pred_label = out.max(dim = 1)[1]            
                        val_acc = utils.acc(pred_label, y_val)

                        ####################
                        # Save weights
                        ####################
                        save_perform_err = val_loss.detach().item()
                        save_perform_acc = val_acc
                        if save_perform_err <= best_test_err:
                            early_stopping = 0
                            best_test_err = save_perform_err
                            torch.save(model.state_dict(), log_path + '/model_err'+str(i)+current_params+'.t7')
                        if save_perform_acc >= best_test_acc:
                            #early_stopping = 0
                            best_test_acc = save_perform_acc
                            torch.save(model.state_dict(), log_path + '/model_acc'+str(i)+current_params+'.t7')
                        else:
                            early_stopping += 1
                    models.append(model)
                torch.cuda.empty_cache()

    err_model_best_average_loss = 100000000000.0
    best_error_model_params = ''
    best_error_model_num_filter = 0
    best_error_model_layer = 0
    best_error_model_lr = 0
    best_error_model_log = ''

    acc_model_best_average_acc = 0.0
    best_acc_model_params = ''
    best_acc_model_num_filter = 0
    best_acc_model_layer = 0
    best_acc_model_lr = 0
    best_acc_model_log = ''

    it_epochs = -1
    best_err_model_epochs_idx = -1
    best_acc_model_epochs_idx = -1
    for lr in [0.001, 0.005, 0.01, 0.05]:
        for num_filter in [16 , 32, 64]:
            for layer in [2, 4, 8]:
                current_params = 'lr_' + str(lr) + '_num_filter_' + str(num_filter) + '_layer_' + str(layer)

                i_validation_error_model_acc = [0.0 for _ in range(10)]
                i_validation_error_model_loss = [0.0 for _ in range(10)]
                i_validation_acc_model_acc = [0.0 for _ in range(10)]
                i_validation_acc_model_loss = [0.0 for _ in range(10)]
                for i in range(10):
                    it_epochs += 1
                    edges = datasets[i]['graph']
                    f_node, e_node = edges[0], edges[1]

                    X_real = utils.in_out_degree(edges, size, weights).to(device)
                    X_img = X_real.clone()

                    y_val   = datasets[i]['val']['label']
                    y_val   = y_val.long().to(device)
                    val_index = datasets[i]['val']['edges'].to(device)

                    edge_index, norm_real, norm_imag = utils.process_magnetic_laplacian(edge_index=edges, gcn=True, net_flow=True,\
                                                            x_real=X_real, edge_weight=weights, normalization = 'sym', return_lambda_max = False)
                    
                    model = SigMaNet_link_prediction_one_laplacian(K=1, num_features=2, hidden=num_filter, label_dim=num_class_link,\
                            i_complex = False,  layer=layer, follow_math=True, gcn=True, net_flow=True, unwind=True, edge_index=edge_index,\
                            norm_real=norm_real, norm_imag=norm_imag).to(device)
                    model = model.to(device)
                    model.load_state_dict(torch.load(log_path + '/model_err'+str(i)+current_params+'.t7'))
                    model.eval()
                    out = model(X_real, X_img, val_index)
                    pred_label = out.max(dim = 1)[1]
                    i_validation_error_model_acc[i] = utils.acc(pred_label, y_val)
                    i_validation_error_model_loss[i] = torch.nn.functional.nll_loss(out, y_val).detach().item()

                    model = SigMaNet_link_prediction_one_laplacian(K=1, num_features=2, hidden=num_filter, label_dim=num_class_link,\
                            i_complex = False,  layer=layer, follow_math=True, gcn=True, net_flow=True, unwind=True, edge_index=edge_index,\
                            norm_real=norm_real, norm_imag=norm_imag).to(device)    
                    model = model.to(device)
                    model.load_state_dict(torch.load(log_path + '/model_acc'+str(i)+current_params+'.t7'))
                    model.eval()
                    out = model(X_real, X_img, val_index)
                    pred_label = out.max(dim = 1)[1]
                    i_validation_acc_model_acc[i] = utils.acc(pred_label, y_val)
                    i_validation_acc_model_loss[i] = torch.nn.functional.nll_loss(out, y_val).detach().item()

                if sum(i_validation_error_model_loss) / 10 < err_model_best_average_loss:
                    best_err_model_epochs_idx = it_epochs - 9
                    err_model_best_average_loss = sum(i_validation_error_model_loss) / 10
                    best_error_model_params = current_params
                    best_error_model_num_filter = num_filter
                    best_error_model_layer = layer
                    best_error_model_lr = lr
                    best_error_model_log += 'i-th split validation loss: '
                    for i in range(10):
                        log = ('{i}: {val_err_loss:.4f}')
                        log = log.format(i=i, val_err_loss=i_validation_error_model_loss[i])
                        best_error_model_log += log + ' '
                    best_error_model_log += '\n i-th split validation acc:  '
                    for i in range(10):
                        log = ('{i}: {val_err_acc:.4f}')
                        log = log.format(i=i, val_err_acc=i_validation_error_model_acc[i])
                        best_error_model_log += log + ' '
                    avg_acc_err_model = sum(i_validation_error_model_acc) / 10
                    best_error_model_log += '\nwith average loss ' + str(err_model_best_average_loss) + 'and average acc ' + str(avg_acc_err_model)
                    best_error_model_log += '\n##########################################################################################\n'
                if sum(i_validation_acc_model_acc) / 10 > acc_model_best_average_acc:
                    best_acc_model_epochs_idx = it_epochs - 9
                    acc_model_best_average_acc = sum(i_validation_acc_model_acc) / 10
                    best_acc_model_params = current_params
                    best_acc_model_num_filter = num_filter
                    best_acc_model_layer = layer
                    best_acc_model_lr = lr
                    best_acc_model_log += 'i-th split validation loss: '
                    for i in range(10):
                        log = ('{i}: {val_acc_loss:.4f}')
                        log = log.format(i=i, val_acc_loss=i_validation_acc_model_loss[i])
                        best_acc_model_log += log + ' '
                    best_acc_model_log += '\ni-th split validation acc:  '
                    for i in range(10):
                        log = ('{i}: {val_acc_acc:.4f}')
                        log = log.format(i=i, val_acc_acc=i_validation_acc_model_acc[i])
                        best_acc_model_log += log + ' '
                    avg_loss_acc_model = sum(i_validation_acc_model_loss) / 10
                    best_acc_model_log += '\nwith average loss ' + str(avg_loss_acc_model) + ' and average acc ' + str(acc_model_best_average_acc)
                    best_acc_model_log += '\n##########################################################################################\n'

    with open(log_path + '/best_error_model_validation_search_log'+'.csv', 'w') as file:
        file.write(best_error_model_log)
        file.write('\n')
    with open(log_path + '/best_acc_model_validation_search_log'+'.csv', 'w') as file:
        file.write(best_acc_model_log)
        file.write('\n')

    log_testing_err_overall = ['' for _ in range(10)]
    log_testing_acc_overall = ['' for _ in range(10)]
    for i in range(10):
        edges = datasets[i]['graph']        
                        
        X_real = utils.in_out_degree(edges, size,  datasets[i]['weights'] ).to(device)
        X_img = X_real.clone()

        y_val   = datasets[i]['val']['label']
        y_test  = datasets[i]['test']['label']
        y_val   = y_val.long().to(device)
        y_test  = y_test.long().to(device)

        val_index = datasets[i]['val']['edges'].to(device)
        test_index = datasets[i]['test']['edges'].to(device)

        edge_index, norm_real, norm_imag = utils.process_magnetic_laplacian(edge_index=edges, gcn=True, net_flow=True,\
                                                            x_real=X_real, edge_weight=weights, normalization = 'sym', return_lambda_max = False)

        model = SigMaNet_link_prediction_one_laplacian(K=1, num_features=2, hidden=best_error_model_num_filter, label_dim=num_class_link,\
                            i_complex = False,  layer=best_error_model_layer, follow_math=True, gcn=True, net_flow=True, unwind=True, edge_index=edge_index,\
                            norm_real=norm_real, norm_imag=norm_imag).to(device)
                               
        model = model.to(device)
        model.load_state_dict(torch.load(log_path + '/model_err'+str(i)+best_error_model_params+'.t7'))
        model.eval()
        out = model(X_real, X_img, val_index)
        pred_label = out.max(dim = 1)[1]
        val_acc_err = utils.acc(pred_label, y_val)
        out = model(X_real, X_img, test_index)
        pred_label = out.max(dim = 1)[1]
        test_acc_err = utils.acc(pred_label, y_test)
        log_str = ('val_acc_err: {val_acc_err:.4f}, '+'test_acc_err: {test_acc_err:.4f}, ')
        log_testing_err_overall[i] += best_error_model_params + '\n'
        log_testing_err_overall[i] += log_str.format(val_acc_err = val_acc_err, test_acc_err = test_acc_err)
        log_testing_err_overall[i] += '\nepochs: '
        log_testing_err_overall[i] += str(nb_epochs[best_err_model_epochs_idx + i] + 1)
        with open(log_path + '/log_testing_err_overall'+str(i)+'.csv', 'w') as file:
            file.write(log_testing_err_overall[i])
            file.write('\n')


        model = SigMaNet_link_prediction_one_laplacian(K=1, num_features=2, hidden=best_acc_model_num_filter, label_dim=num_class_link,\
                            i_complex = False,  layer=best_acc_model_layer, follow_math=True, gcn=True, net_flow=True, unwind=True, edge_index=edge_index,\
                            norm_real=norm_real, norm_imag=norm_imag).to(device)
        model = model.to(device)
        model.load_state_dict(torch.load(log_path + '/model_acc'+str(i)+best_acc_model_params+'.t7'))
        model.eval()
        out = model(X_real, X_img, val_index)
        pred_label = out.max(dim = 1)[1]
        val_acc = utils.acc(pred_label, y_val)
        out = model(X_real, X_img, test_index)
        pred_label = out.max(dim = 1)[1]
        test_acc = utils.acc(pred_label, y_test)
        log_str = ('val_acc: {val_acc:.4f}, '+'test_acc: {test_acc:.4f}, ')
        log_testing_acc_overall[i] += best_acc_model_params + '\n'
        log_testing_acc_overall[i] += log_str.format(val_acc = val_acc, test_acc = test_acc)
        log_testing_acc_overall[i] += '\nepochs: '
        log_testing_acc_overall[i] += str(nb_epochs[best_acc_model_epochs_idx + i] + 1)
        with open(log_path + '/log_testing_acc_overall'+str(i)+'.csv', 'w') as file:
            file.write(log_testing_acc_overall[i])
            file.write('\n')