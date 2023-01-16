import numpy as np
import pandas as pd
import tensorly as tl  # Note the effect of different unfolding or folding methods on tensor construction.


def ST_LRST(Y, delta, LA, p, rho, lamda, alpha, beta, gama, phi):
    '''
    :param Y: traffic data tensor
    :param delta: Toeplitz matrix
    :param LA: Laplacian regularizer
    :param p, lamda: weight parameters
    :param rho, alpha, beta, gama, phi: penalty parameters
    :return: traffic normal tensor M, abnormal tensor S
    '''
    I1, I2, I3 = np.shape(Y)
    dim = [I1, I2, I3]
    # R = np.linalg.matrix_rank(M)
    R = 5

    L = np.zeros(np.insert(dim, 0, len(dim)))
    for k in range(len(dim)):
        L[k] = Y
    # L = np.broadcast_to(np.copy(Y), np.insert(dim, 0, len(dim)))
    M = np.copy(Y)
    S = np.zeros(dim)
    W = np.zeros(dim)
    V = np.zeros(dim)
    Z = tl.fold(np.dot(delta, tl.unfold(W, mode=1)), mode=1, shape=dim)
    X = np.copy(Y)

    # Initialization
    tol = 1e-3
    Iter = 0
    Max_iter = 200
    T = np.zeros(np.insert(dim, 0, len(dim)))
    Y1 = np.zeros(dim)
    Y2 = np.zeros(dim)
    Y3 = np.zeros(dim)
    Y4 = np.zeros(dim)

    for t in range(Max_iter):
        # Update L
        for k in range(3):
            UU, SS, VV = np.linalg.svd(tl.unfold(L[k], k), full_matrices=None)
            A = UU[:, 0:R]
            B = VV[0:R, :].T
            L[k] = tl.fold(
                D_shrinke(tl.unfold(M, k) + (p[k] * np.dot(A, B.T) - tl.unfold(T[k], k)) / rho[k], p[k] / rho[k]),
                k, dim)
        # L_hat = np.einsum('k, kmnt -> mnt', p, L)

        # Update S
        H = (alpha * (X - M - Y1 / alpha) + beta * (W + Y2 / beta) + phi * (V + Y4 / phi)) / (alpha + beta + phi)
        S = np.multiply(np.sign(H), np.maximum(np.abs(H) - lamda[0] / (alpha + beta + phi), 0))

        # Update W
        A = beta * (tl.unfold(S, 1) - tl.unfold(Y2, 1) / beta) + gama * np.dot(delta.T,
                                                                               (tl.unfold(Z, 1) + tl.unfold(Y3,
                                                                                                            1) / gama))
        W = tl.fold(np.dot(np.linalg.inv(beta * np.eye(I2) + gama * np.dot(delta.T, delta)), A), mode=1, shape=dim)

        # Update Z
        J = tl.fold(np.dot(delta, tl.unfold(W, mode=1)), mode=1, shape=dim) - Y3 / gama
        Z = np.multiply(np.sign(J), np.maximum(np.abs(J) - lamda[1] / gama, 0))

        # Update V
        B_mat = 2 * lamda[2] * LA + phi * np.eye(I1)
        V_mat = np.dot(np.linalg.inv(B_mat), (phi * tl.unfold(S, mode=0) - tl.unfold(Y4, mode=0)))
        V = tl.fold(V_mat, mode=0, shape=dim)

        # Update M
        M = ((rho[0] * L[0] + rho[1] * L[1] + rho[2] * L[2] + T[0] + T[1] + T[2]) + alpha * (
                X - S - Y1 / alpha)) / (np.sum(rho) + alpha)
        X0 = M + S

        Iter += 1
        Error_mat = tl.norm(X - X0) / tl.norm(X)
        print(Iter, 'ERROR:', Error_mat)

        if Error_mat < tol or Iter >= Max_iter:
            break

        # update dual variables and parameters
        T = T + np.mean(rho) * (L - np.broadcast_to(M, np.insert(dim, 0, len(dim))))
        Y1 = Y1 + alpha * (M + S - X)
        Y2 = Y2 + beta * (W - S)
        Y3 = Y3 + np.mean(rho) * (Z - tl.fold(np.dot(delta, tl.unfold(W, mode=1)), mode=1, shape=dim))
        Y4 = Y4 + phi * (V - S)
        rho = np.minimum(np.dot(1.05, rho), 1e5 * np.ones(3))
        alpha = np.minimum(np.dot(1.05, alpha), 1e5)
        beta = np.minimum(np.dot(1.05, beta), 1e5)
        gama = np.minimum(np.dot(1.05, gama), 1e5)
        phi = np.minimum(np.dot(1.05, phi), 1e5)

    return M, S


def D_shrinke(X, theta):
    U, SS, V = np.linalg.svd(X, full_matrices=0)
    SS = np.maximum(SS - theta, 0)
    D = np.dot(np.dot(U, np.diag(SS)), V)

    return D


def Acc_compute(S_mat, ano_mat, ano_inf, Free_speed):
    '''
    :param S_mat: detected anomalies
    :param ano_mat: true anomalies
    :param ano_inf: anomaly information
    :param Free_speed: free flow speed
    '''
    # Label construction (binary conversion)
    S_mat_bin = np.copy(S_mat)
    ano_mat_bin = np.copy(ano_mat)
    S_mat_bin[np.abs(S_mat_bin) <= 5] = 0
    S_mat_bin[np.abs(S_mat_bin) > 5] = 1
    ano_mat_bin[ano_mat_bin != 0] = 1
    I1, I2 = np.shape(ano_mat_bin)
    I21 = 144

    # Construct a judgment matrix
    ano_TF = np.zeros((I1, I2)).astype(int)
    for i in range(I1):
        for j in range(I2):
            if S_mat_bin[i][j] == 1 and ano_mat_bin[i][j] == 1:  # TP
                ano_TF[i][j] = 11
            elif S_mat_bin[i][j] == 1 and ano_mat_bin[i][j] == 0:  # FP
                ano_TF[i][j] = 22
            elif S_mat_bin[i][j] == 0 and ano_mat_bin[i][j] == 1:  # FN
                ano_TF[i][j] = 33
            elif S_mat_bin[i][j] == 0 and ano_mat_bin[i][j] == 0:  # TN
                ano_TF[i][j] = 44

    TP_num_all = np.count_nonzero(ano_TF == 11)
    FP_num_all = np.count_nonzero(ano_TF == 22)
    FN_num_all = np.count_nonzero(ano_TF == 33)
    # Special case
    if TP_num_all == 0:
        acc_ave_per = 0
        acc_ave_recall = 0
        F1_score = 0
    else:
        acc_ave_per = TP_num_all / (TP_num_all + FP_num_all)
        acc_ave_recall = TP_num_all / (TP_num_all + FN_num_all)
        F1_score = 2 * acc_ave_per * acc_ave_recall / (acc_ave_per + acc_ave_recall + 1e-10)
    print('Precision rate：', acc_ave_per)
    print('Recall rate：', acc_ave_recall)
    print('F1 score: ', F1_score)

    # Determine the ability to detect anomalies
    ano_num, _ = np.shape(ano_inf)
    acc_mat = np.zeros((ano_num, 5))  # for each anomaly: [precision, recall, intensity]
    # ano_inf column=[day, central_location_x-spatial, central_location_y-temporal, Impact_scope-temporal, Impact_scope-spatial, Impact_intensity]
    for i in range(ano_num):
        point_x = ano_inf[i][1]
        point_y = ano_inf[i][2]
        spa_range = ano_inf[i][3]
        tem_range = ano_inf[i][4]

        # Anomaly scope
        range_tem_left = int(max(point_y - tem_range, 0) + ano_inf[i][0] * I21)
        range_tem_right = int(min(point_y + tem_range, I21) + ano_inf[i][0] * I21)

        # Impacted road segments (※※ Rely on topological adjacency)
        str_road = (spa_range.split('[')[1]).split(']')[0]
        spa_roads = np.array(str_road.split(', '))

        # calculate precision and recall
        sub_mat = ano_TF.take([int(i) for i in spa_roads], 0)
        sub_mat = sub_mat[:, range_tem_left:range_tem_right]
        TP_num = np.count_nonzero(sub_mat == 11)
        FP_num = np.count_nonzero(sub_mat == 22)
        FN_num = np.count_nonzero(sub_mat == 33)
        # print(i, TP_num, FP_num, FN_num)
        if TP_num == 0:
            acc_mat[i][0] = 0
            acc_mat[i][1] = 0
        else:
            acc_per = TP_num / (TP_num + FP_num)
            acc_recall = TP_num / (TP_num + FN_num)
            acc_mat[i][0] = acc_per
            acc_mat[i][1] = acc_recall

        # Calculation of the accuracy of impact intensity
        S_submat = S_mat.take([int(i) for i in spa_roads], 0)
        S_submat = S_submat[:, range_tem_left:range_tem_right]
        ano_submat = ano_mat.take([int(i) for i in spa_roads], 0)
        ano_submat = ano_submat[:, range_tem_left:range_tem_right]
        # Average speed
        Ave_submat = ave_mat.take([int(i) for i in spa_roads], 0)
        Ave_submat = Ave_submat[:, range_tem_left:range_tem_right]
        # Free speed
        # Free_submat = Free_speed.take([int(i) for i in spa_roads])
        # Free_submat = np.tile(Free_submat.reshape(-1, 1), (1, range_tem_right - range_tem_left))
        acc_mat[i][3] = np.average(np.abs(np.abs(S_submat) - ano_submat))
        acc_mat[i][4] = np.average(np.abs(np.abs(S_submat) - ano_submat) / ano_submat)
        # Average impact intensity
        # degree_imp = np.average(np.abs(S_submat) / Free_submat)
        degree_imp = np.average(np.abs(S_submat) / Ave_submat)

        # Within the true value range? (ano_inf[i][5])
        if ano_inf[i][5] == 0.4 and degree_imp > 0.3 and degree_imp < 0.5:
            acc_mat[i][2] = 99  # Yes, return an identifier
        elif ano_inf[i][5] == 0.55 and degree_imp > 0.5 and degree_imp < 0.6:
            acc_mat[i][2] = 99
        elif ano_inf[i][5] == 0.65 and degree_imp > 0.6 and degree_imp < 0.7:
            acc_mat[i][2] = 99
        else:
            # No, output deviation percentage
            acc_mat[i][2] = abs(degree_imp - ano_inf[i][5])

    acc_intensity=len(np.where(acc_mat[:,2]==99)[0])/len(acc_mat[:,2])
    RMSE_Anomaly=np.average(acc_mat[:,3])
    MAPE_Anomaly=np.average(acc_mat[:,4])
    print('Acc_Anomaly：', acc_intensity)
    print('RMSE_Anomaly：',RMSE_Anomaly)
    print('MAPE_Anomaly：',MAPE_Anomaly)

    return acc_mat


if __name__ == '__main__':
    file_name = 'Spatial_syn_mat_mix.csv'
    data = np.array(pd.read_csv(file_name, header=None))
    data_tensor = tl.fold(data, mode=0, shape=(data.shape[0], 144, 30))
    I1, I2, I3 = np.shape(data_tensor)

    # S2
    delta = np.eye(I2) - np.eye(I2, k=1)
    delta[I2 - 1, 0] = -1
    # S1
    filename2 = '../AA_Xian.csv'
    AA = np.array(pd.read_csv(filename2, header=None))
    DU = np.diag(np.sum(AA, axis=1))  # By row
    LA = DU - AA  # Laplacian matrix

    p = 10 * np.ones(3) / 3
    rho = 1e-3 * np.ones(3)
    lamda = [1e-2, 1e-1, 1e-3]
    alpha = 1e-3
    beta = 1e-3
    gama = 1e-1
    phi = 1e-5
    M, S = ST_LRST(data_tensor, delta, LA, p, rho, lamda, alpha, beta, gama, phi)
    np.savetxt('Ano_mat_LRST_mix.csv', tl.unfold(S, mode=0), fmt='%.2f', delimiter=',')

    # Accuracy assessment
    filename3 = 'Spatial_anomaly_mat_mix.csv'
    ano_mat = np.array(pd.read_csv(filename3, header=None))
    # Anomaly information
    filename4 = 'Spatial_anomaly_infor_mix.csv'
    ano_inf = np.array(pd.read_csv(filename4))
    # Average speed(normal)
    filename5 = 'Spatial_ave_mat.csv'
    ave_mat = np.array(pd.read_csv(filename5, header=None))
    # Free speed
    filename6 = '../Free_speed.csv'
    Free_speed = np.array(pd.read_csv(filename6))[:, 2]

    acc_mat = Acc_compute(tl.unfold(S, mode=0), ano_mat, ano_inf, Free_speed)  # S为异常【矩阵】
    np.savetxt('Acc_mat_LRST_mix.csv', acc_mat, fmt='%.3f', delimiter=',')

    print()
