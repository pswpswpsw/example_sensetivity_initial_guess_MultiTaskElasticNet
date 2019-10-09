import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import ElasticNet, enet_path, lasso_path, MultiTaskElasticNet, Lasso
import sklearn


def compute_kou_index(true_tj, true_eigenTj):
    # true_tj = M, N
    # true_eigenT = M,L
    # B = L, N

    # 1. compute koopman modes
    B = np.linalg.lstsq(true_eigenTj, true_tj)[0]

    # 2. normalize koopman modes
    # B_normalized = np.matmul(np.linalg.inv(np.diag(np.linalg.norm(B, axis=1))), B)

    # 3. recompute the eigenfunctions
    true_eigenTj_new = np.matmul(true_eigenTj, np.diag(np.linalg.norm(B, axis=1)))

    # 4. compute the sum of absolute value
    abs_sum_of_eigen_function = np.sum(abs(true_eigenTj_new), axis=0)

    index_chosen = np.argsort(abs_sum_of_eigen_function)[::-1]  # best modes comes first
    abs_sum_of_index_chosen = np.sort(abs_sum_of_eigen_function)[::-1]

    # 5. bouns: compute the normalized residual remaining
    relative_rec_error_from_top_i_rank_1_sum_list = []
    for i in range(len(index_chosen)):
        top_i_index = index_chosen[:i + 1]
        X_rec_from_top_i = np.matmul(true_eigenTj[:, top_i_index], B[top_i_index, :])
        relative_rec_error_from_top_i_rank_1_sum_list.append(
            np.linalg.norm(X_rec_from_top_i - true_tj) / np.linalg.norm(true_tj))

    return index_chosen, abs_sum_of_index_chosen, np.array(relative_rec_error_from_top_i_rank_1_sum_list)

# prepare data
data = np.load('data.npz')
A = data['A']
B = data['B']

# print(A.shape) # (1600,14)
# print(B.shape) # (1600,2)

# note that data is centered so no need to fit the bias
# print(A.mean(axis=0))
# print(B.mean(axis=0))

# zhang's problem

A = np.array([[1,-1e3,1e3]])
B = np.array([[1,0,1e-3]])
KM = np.array([[1,0,0],
               [0,1,0],
               [0,1,1e-6]])




assert np.linalg.norm(np.matmul(A,KM)-B) < 1e-14

# modified zhang's problem
A_list = [A]
B_list = [B]
A_next = np.copy(A)
B_next = np.copy(B)
total_time_step = 20 # long time
# total_time_step = 3 # short time

for i in range(total_time_step):
    A_next = np.matmul(A_next, np.diag([0.1,0.9,0.1])) # updating the Koopman eigensignals
    B_next = np.matmul(A_next, KM)
    A_list.append(A_next)
    B_list.append(B_next)

A = np.vstack(A_list)
B = np.vstack(B_list)

print(A.shape) # (1600,14)
print(B.shape) # (1600,2)


num_alpha = 100
tol = 1e-12
l1_ratio = 1
max_iter = 1e4

disable_single_enet = True

# case 1: performing enet path by calling ElasticNet individually
alpha_array = np.logspace(-16,1,num_alpha)
alpha_array = alpha_array[::-1]
coef_list_1 = []
coef_list_2 = []
coef_list_1_m = []
coef_list_2_m = []
coef_list_1_m_w = []
coef_list_2_m_w = []
coef_list_1_l = []
coef_list_2_l = []


# case 2: run enet_path to do the same thing

alphas_enet, coefs_enet, _ = enet_path(A, B, max_iter=max_iter, tol=tol,
                                       alphas=np.logspace(-16,1,num_alpha),
                                       l1_ratio=l1_ratio, fit_intercept=False)


# compute residuals, MSE

residual_list = []
for i in range(num_alpha):
    residual = sklearn.metrics.mean_squared_error(B, np.matmul(A, coefs_enet[:,:,i].T))
    residual_list.append(residual)

residual_array = np.array(residual_list)



coef_mag_list = []
for i in range(num_alpha):
    coef_mag_list.append(np.linalg.norm(coefs_enet[:,:,i].T))

plt.figure()
plt.plot(np.log10(alphas_enet), coef_mag_list,'b-*')
plt.show()


plt.figure()
plt.semilogy(np.log10(alphas_enet), residual_array,'r-*')
plt.show()


# compute kou's criterion
index, energy, residualsss = compute_kou_index(B,A)

# for each column in X, compute the total energy
print('kou index top 3 order= ', index)
print('kou energy top 3 value = ', energy)



print('A =',A)
print('B =',B)


B_rec_only_1 = np.matmul(A[:,1:2], KM[1:2,:])
print('B_rec = ',B_rec_only_1 )

plt.figure()
plt.plot(B_rec_only_1,'go-')
plt.plot(B,'b^--')
plt.ylim([-5,2])
plt.show()

print('True KM = \n', KM)

index = 2
# index = 29 # for 50 timestep
print('my algorithm find solution = \n', coefs_enet[:,:,index].T, '\n alpha = ', alphas_enet[index])

B_rec_lasso = np.matmul(A[:,:], coefs_enet[:,:,index].T[:,:])
plt.figure()
plt.plot(B_rec_lasso,'ro-')
plt.plot(B,'b^--')
plt.ylim([-5,2])
plt.show()



# print('my algorithm find solution = \n', coefs_enet[:,:,4].T, ' alpha = ', alphas_enet[4])
# print('my algorithm find solution = \n', coefs_enet[:,:,10].T, ' alpha = ', alphas_enet[10])
