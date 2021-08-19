import os
import numpy as np
import torch
import torch.utils.data
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from .metrics import masked_mape_np
from scipy.sparse.linalg import eigs
import csv


def re_normalization(x, mean, std):
    x = x * std + mean
    return x


def get_adjacency_matrix(distance_df_filename, num_of_vertices):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    with open(distance_df_filename, 'r') as f:
        reader = csv.reader(f)
        header = f.__next__()
        edges = [(int(i[0]), int(i[1])) for i in reader]

    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                 dtype=np.float32)

    for i, j in edges:
        A[i, j] = 1

    return A


def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))

    L = D - W

    lambda_max = eigs(L, k=1, which='LR')[0].real

    return (2 * L) / lambda_max - np.identity(W.shape[0])


def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)

    K: the maximum order of chebyshev polynomials

    Returns
    ----------
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}

    '''

    N = L_tilde.shape[0]

    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials


def load_graphdata_channel1(graph_signal_matrix_filename, num_of_hours, num_of_days, num_of_weeks, DEVICE, batch_size, shuffle=True):
    '''
    这个是为PEMS的数据准备的函数
    将x,y都处理成归一化到[-1,1]之前的数据;
    每个样本同时包含所有监测点的数据，所以本函数构造的数据输入时空序列预测模型；
    该函数会把hour, day, week的时间串起来；
    注： 从文件读入的数据，x是最大最小归一化的，但是y是真实值
    这个函数转为mstgcn，astgcn设计，返回的数据x都是通过减均值除方差进行归一化的，y都是真实值
    :param graph_signal_matrix_filename: str
    :param num_of_hours: int
    :param num_of_days: int
    :param num_of_weeks: int
    :param DEVICE:
    :param batch_size: int
    :return:
    three DataLoaders, each dataloader contains:
    test_x_tensor: (B, N_nodes, in_feature, T_input)
    test_decoder_input_tensor: (B, N_nodes, T_output)
    test_target_tensor: (B, N_nodes, T_output)

    '''

    file = os.path.basename(graph_signal_matrix_filename).split('.')[0]

    dirpath = os.path.dirname(graph_signal_matrix_filename)

    filename = os.path.join(dirpath,
                            file + '_r' + str(num_of_hours) + '_d' + str(num_of_days) + '_w' + str(num_of_weeks)) +'_astcgn'

    print('load file:', filename)

    file_data = np.load(filename + '.npz')

    train_week = file_data['train_week']  # (8979, 307, 3, 12)
    train_week = train_week[:, :, 0:1, :]  # (8979, 307, 1, 12)
    train_day = file_data['train_day']
    train_day = train_day[:, :, 0:1, :]
    train_recent = file_data['train_recent']
    train_recent = train_recent[:, :, 0:1, :]
    train_target = file_data['train_target']

    val_week = file_data['val_week']
    val_week = val_week[:, :, 0:1, :]
    val_day = file_data['val_day']
    val_day = val_day[:, :, 0:1, :]
    val_recent = file_data['val_recent']
    val_recent = val_recent[:, :, 0:1, :]
    val_target = file_data['val_target']

    test_week = file_data['test_week']
    test_week = test_week[:, :, 0:1, :]
    test_day = file_data['test_day']
    test_day = test_day[:, :, 0:1, :]
    test_recent = file_data['test_recent']
    test_recent = test_recent[:, :, 0:1, :]
    test_target = file_data['test_target']

    # mean = file_data['mean'][:, :, 0:1, :]  # (1, 1, 3, 1)
    # std = file_data['std'][:, :, 0:1, :]  # (1, 1, 3, 1)
    stats_data = {}
    # for type in ['week', 'day', 'recent']:
    #     stats_data[type + '_mean'] = file_data['stats_' + type]['_mean']
    #     stats_data[type + '_std'] = file_data['stats_' + type]['_std']

    # ------- train_loader -------
    train_week_tensor = torch.from_numpy(train_week).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    train_day_tensor = torch.from_numpy(train_day).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    train_recent_tensor = torch.from_numpy(train_recent).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    train_dataset = torch.utils.data.TensorDataset(train_week_tensor, train_day_tensor, train_recent_tensor,
                                                   train_target_tensor)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    # ------- val_loader -------
    val_week_tensor = torch.from_numpy(val_week).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    val_day_tensor = torch.from_numpy(val_day).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    val_recent_tensor = torch.from_numpy(val_recent).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    val_target_tensor = torch.from_numpy(val_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    val_dataset = torch.utils.data.TensorDataset(val_week_tensor, val_day_tensor, val_recent_tensor, val_target_tensor)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ------- test_loader -------
    test_week_tensor = torch.from_numpy(test_week).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    test_day_tensor = torch.from_numpy(test_day).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    test_recent_tensor = torch.from_numpy(test_recent).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    test_target_tensor = torch.from_numpy(test_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    test_dataset = torch.utils.data.TensorDataset(test_week_tensor, test_day_tensor, test_recent_tensor,
                                                  test_target_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # print
    print('train:', train_week_tensor.size(), train_day_tensor.size(), train_recent_tensor.size(),
          train_target_tensor.size())
    print('val:', val_week_tensor.size(), val_day_tensor.size(), val_recent_tensor.size(), val_target_tensor.size())
    print('test:', test_week_tensor.size(), test_day_tensor.size(), test_recent_tensor.size(),
          test_target_tensor.size())

    return train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, stats_data


def compute_val_loss_mstgcn(net, val_loader, criterion, sw, epoch, limit=None):
    '''
    for rnn, compute mean loss on validation set
    :param net: model
    :param val_loader: torch.utils.data.utils.DataLoader
    :param criterion: torch.nn.MSELoss
    :param sw: tensorboardX.SummaryWriter
    :param global_step: int, current global_step
    :param limit: int,
    :return: val_loss
    '''

    net.train(False)  # ensure dropout layers are in evaluation mode

    # 暂时不追踪网络参数中的导数
    with torch.no_grad():

        val_loader_length = len(val_loader)  # nb of batch

        tmp = []  # 记录了所有batch的loss

        for batch_index, batch_data in enumerate(val_loader):
            val_w, val_d, val_r, labels = batch_data
            # outputs = net(val_r)
            outputs = net([val_w, val_d, val_r])
            loss = criterion(outputs, labels)  # 计算误差
            tmp.append(loss.item())
            print('validation batch %s / %s, loss: %.2f' % (batch_index + 1, val_loader_length, loss.item()))
            if (limit is not None) and batch_index >= limit:
                break

        validation_loss = sum(tmp) / len(tmp)
        sw.add_scalar('validation_loss', validation_loss, epoch)
    return validation_loss

def evaluate_on_test_mstgcn(net, test_loader, test_target_tensor, sw, epoch, _mean, _std):
    '''
    for rnn, compute MAE, RMSE, MAPE scores of the prediction for every time step on testing set.
    :param net: model
    :param test_loader: torch.utils.data.utils.DataLoader
    :param test_target_tensor: torch.tensor (B, N_nodes, T_output, out_feature)=(B, N_nodes, T_output, 1)
    :param sw:
    :param epoch: int, current epoch
    :param _mean: (1, 1, 3(features), 1)
    :param _std: (1, 1, 3(features), 1)
    '''

    net.train(False)  # ensure dropout layers are in test mode

    with torch.no_grad():

        test_loader_length = len(test_loader)

        test_target_tensor = test_target_tensor.cpu().numpy()

        prediction = []  # 存储所有batch的output

        for batch_index, batch_data in enumerate(test_loader):

            encoder_inputs, labels = batch_data

            outputs = net(encoder_inputs)

            prediction.append(outputs.detach().cpu().numpy())

            if batch_index % 100 == 0:
                print('predicting testing set batch %s / %s' % (batch_index + 1, test_loader_length))

        prediction = np.concatenate(prediction, 0)  # (batch, T', 1)
        prediction_length = prediction.shape[2]

        for i in range(prediction_length):
            assert test_target_tensor.shape[0] == prediction.shape[0]
            print('current epoch: %s, predict %s points' % (epoch, i))
            mae = mean_absolute_error(test_target_tensor[:, :, i], prediction[:, :, i])
            rmse = mean_squared_error(test_target_tensor[:, :, i], prediction[:, :, i]) ** 0.5
            mape = masked_mape_np(test_target_tensor[:, :, i], prediction[:, :, i], 0)
            print('MAE: %.2f' % (mae))
            print('RMSE: %.2f' % (rmse))
            print('MAPE: %.2f' % (mape))
            print()
            if sw:
                sw.add_scalar('MAE_%s_points' % (i), mae, epoch)
                sw.add_scalar('RMSE_%s_points' % (i), rmse, epoch)
                sw.add_scalar('MAPE_%s_points' % (i), mape, epoch)

def predict_and_save_results_mstgcn(net, data_loader, data_target_tensor, num_of_vertices, global_step, stats_data, params_path, type):
    """
    :param net: nn.Module
    :param data_loader: torch.utils.data.utils.DataLoader
    :param data_target_tensor: tensor
    :param epoch: int
    :param _mean: (1, 1, 3, 1)
    :param _std: (1, 1, 3, 1)
    :param params_path: the path for saving the results
    :return:
    """
    net.train(False)  # ensure dropout layers are in test mode

    with torch.no_grad():

        data_target_tensor = data_target_tensor.cpu().numpy()

        loader_length = len(data_loader)  # nb of batch

        prediction = []  # 存储所有batch的output

        # input = []  # 存储所有batch的input

        for batch_index, batch_data in enumerate(data_loader):

            # encoder_inputs, labels = batch_data
            data_w, data_h, data_r, labels = batch_data

            # input.append(encoder_inputs[:, :, 0:1].cpu().numpy())  # (batch, T', 1)
            # input.append(data_w[:, :, 0:1].cpu().numpy())
            # outputs = net(data_r)
            outputs = net([data_w, data_h, data_r])
            prediction.append(outputs.detach().cpu().numpy())

            print('predicting data set batch %s / %s' % (batch_index + 1, loader_length))

        # input = np.concatenate(input, 0)
        # mean = stats_data['recent_mean']
        # std = stats_data['recent_std']

        # input = re_normalization(input, mean, std)

        prediction = np.concatenate(prediction, 0)  # (batch, T', 1)

        # print('input:', input.shape)
        print('prediction:', prediction.shape)
        print('data_target_tensor:', data_target_tensor.shape)
        output_filename = os.path.join(params_path, 'output_epoch_%s_%s' % (global_step, type))
        # np.savez(output_filename, input=input, prediction=prediction, data_target_tensor=data_target_tensor)
        np.savez(output_filename, prediction=prediction, data_target_tensor=data_target_tensor)

        # 计算误差
        excel_list = []
        prediction_length = prediction.shape[2]

        for i in range(prediction_length):
            assert data_target_tensor.shape[0] == prediction.shape[0]
            print('current epoch: %s, predict %s points' % (global_step, i))
            mae = mean_absolute_error(data_target_tensor[:, :, i], prediction[:, :, i])
            rmse = mean_squared_error(data_target_tensor[:, :, i], prediction[:, :, i]) ** 0.5
            mape = masked_mape_np(data_target_tensor[:, :, i], prediction[:, :, i], 0)
            print('MAE: %.2f' % (mae))
            print('RMSE: %.2f' % (rmse))
            print('MAPE: %.2f' % (mape))
            excel_list.extend([mae, rmse, mape])

        # print overall results
        mae = mean_absolute_error(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1))
        rmse = mean_squared_error(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1)) ** 0.5
        mape = masked_mape_np(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0)
        print('all MAE: %.2f' % (mae))
        print('all RMSE: %.2f' % (rmse))
        print('all MAPE: %.2f' % (mape))

        excel_list.extend([mae, rmse, mape])
        print(excel_list)


