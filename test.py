import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import ElasticNet, enet_path, lasso_path, MultiTaskElasticNet, Lasso
import sklearn

# prepare data
data = np.load('data.npz')
A = data['A']
B = data['B']

print A.shape
print B.shape

# note that data is centered so no need to fit the bias
print A.mean(axis=0)
print B.mean(axis=0)

num_alpha = 100
tol = 1e-8
l1_ratio = 0.5
max_iter = 3000

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


for index, alpha in enumerate(alpha_array):

    print alpha

    if not disable_single_enet:
        clf = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, tol=tol, max_iter=max_iter, fit_intercept=False)
        clf.fit(A, B)
        coef_list_1.append(clf.coef_[0])
        coef_list_2.append(clf.coef_[1])

    clf_m_wrong = MultiTaskElasticNet(alpha=alpha, l1_ratio=l1_ratio, tol=tol, max_iter=max_iter, fit_intercept=False)
    clf_m_wrong.fit(A, B)

    if index == 0:
        clf_m = MultiTaskElasticNet(alpha=alpha, l1_ratio=l1_ratio, tol=tol, max_iter=max_iter, fit_intercept=False)
        clf_m.fit(A, B)
    else:
        clf_m.warm_start = True
        clf_m.alpha = alpha
        clf_m.fit(A, B)

    coef_copy_wrong = np.copy(clf_m_wrong.coef_)
    coef_copy = np.copy(clf_m.coef_)

    coef_list_1_m.append(coef_copy[0])
    coef_list_2_m.append(coef_copy[1])
    coef_list_1_m_w.append(coef_copy_wrong[0])
    coef_list_2_m_w.append(coef_copy_wrong[1])



if not disable_single_enet:
    # enet
    plt.figure()
    plt.ylabel('coef')
    plt.title('ElasticNet with single target: first one')
    plt.plot(-np.log10(alpha_array), coef_list_1)
    plt.xlabel('-log10(alpha)')
    plt.savefig('single_run_enet_coef_1.png')
    plt.close()

    plt.figure()
    plt.ylabel('coef')
    plt.title('ElasticNet with single target: second one')
    plt.plot(-np.log10(alpha_array), coef_list_2)
    plt.xlabel('-log10(alpha)')
    plt.savefig('single_run_enet_coef_2.png')
    plt.close()

# m_enet
plt.figure()
plt.ylabel('coef')
plt.title('With Warm-up MultiTaskELasticNet: first one')
plt.plot(-np.log10(alpha_array), coef_list_1_m)
plt.xlabel('-log10(alpha)')
plt.savefig('single_run_multi_enet_coef_1_reuse.png')
plt.close()

plt.figure()
plt.title('With Warm-up MultiTaskElasticNet: second one')
plt.ylabel('coef')
plt.plot(-np.log10(alpha_array), coef_list_2_m)
plt.xlabel('-log10(alpha)')
plt.savefig('single_run_multi_enet_coef_2_reuse.png')
plt.close()

# m_enet: but wrongly setup, i.e., no warm-up
plt.figure()
plt.ylabel('coef')
plt.title('Without Warm-up MultiTaskELasticNet: first one')
plt.plot(-np.log10(alpha_array), coef_list_1_m_w)
plt.xlabel('-log10(alpha)')
plt.savefig('single_run_multi_enet_coef_1_no_reuse.png')
plt.close()

plt.figure()
plt.title('Without Warm-up MultiTaskElasticNet: second one')
plt.ylabel('coef')
plt.plot(-np.log10(alpha_array), coef_list_2_m_w)
plt.xlabel('-log10(alpha)')
plt.savefig('single_run_multi_enet_coef_2_no_reuse.png')
plt.close()



# case 2: run enet_path to do the same thing

alphas_enet, coefs_enet, _ = enet_path(A, B, max_iter=max_iter, tol=tol,
                                       alphas=np.logspace(-16,1,num_alpha),
                                       l1_ratio=l1_ratio, fit_intercept=False)


# compute residuals, MSE

residual_list = []
for i in xrange(num_alpha):
    residual = sklearn.metrics.mean_squared_error(B, np.matmul(A, coefs_enet[:,:,i].T))
    residual_list.append(residual)

residual_array = np.array(residual_list)

plt.figure()
plt.ylabel('coef')
plt.title('Direct calling enet_path')
plt.plot(-np.log10(alpha_array), coefs_enet[0,:,:].T)
plt.xlabel('-log10(alpha)')
plt.savefig('enet_path_coef_1.png')
plt.close()

plt.figure('Direct calling enet_path')
plt.ylabel('coef')
plt.plot(-np.log10(alpha_array), coefs_enet[1,:,:].T)
plt.xlabel('-log10(alpha)')
plt.savefig('enet_path_coef_2.png')
plt.close()

plt.figure('Direct calling enet_path')
plt.ylabel('MSE')
plt.semilogy(-np.log10(alpha_array), residual_array)
plt.xlabel('-log10(alpha)')
plt.savefig('enet_path_residual.png')
plt.close()







