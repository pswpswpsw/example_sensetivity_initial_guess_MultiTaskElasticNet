import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import ElasticNet, enet_path, lasso_path, MultiTaskElasticNet, Lasso


# prepare data
data = np.load('data.npz')
A = data['A']
B = data['B']

print A.shape
print B.shape

# note that data is centered so no need to fit the bias
print A.mean(axis=0)
print B.mean(axis=0)


# case 1: performing enet path by calling ElasticNet individually
alpha_array = np.logspace(-16,1,200)
coef_list_1 = []
coef_list_2 = []
coef_list_1_m = [] 
coef_list_2_m = []
coef_list_1_l = [] 
coef_list_2_l = []


for alpha in alpha_array:
    
    clf = ElasticNet(alpha=alpha, l1_ratio=0.5, tol=1e-6, max_iter=3000, 
                     warm_start=False, random_state=1, fit_intercept=False)
    
    clf_m = MultiTaskElasticNet(alpha=alpha, l1_ratio=0.5, tol=1e-6, max_iter=3000, 
                                warm_start=False, random_state=1, fit_intercept=False)
    # get fitted
    clf_m.fit(A, B)
    clf.fit(A, B)
    
    coef_list_1.append(clf.coef_[0])
    coef_list_2.append(clf.coef_[1])
    coef_list_1_m.append(clf_m.coef_[0])
    coef_list_2_m.append(clf_m.coef_[1])

# enet
plt.figure()
plt.plot(-np.log10(alpha_array), coef_list_1)
plt.xlabel('-log10(alpha)')
plt.savefig('single_run_enet_coef_1.png')
plt.close()

plt.figure()
plt.plot(-np.log10(alpha_array), coef_list_2)
plt.xlabel('-log10(alpha)')
plt.savefig('single_run_enet_coef_2.png')
plt.close()

# m_enet
plt.figure()
plt.plot(-np.log10(alpha_array), coef_list_1_m)
plt.xlabel('-log10(alpha)')
plt.savefig('single_run_multi_enet_coef_1.png')
plt.close()

plt.figure()
plt.plot(-np.log10(alpha_array), coef_list_2_m)
plt.xlabel('-log10(alpha)')
plt.savefig('single_run_multi_enet_coef_2.png')
plt.close()


# case 2: run enet_path to do the same thing

alphas_enet, coefs_enet, _ = enet_path(A, B, max_iter=3000, tol=1e-6, 
                                       alphas=np.logspace(-16,1,200),precompute=False,
                                       l1_ratio=0.5, random_state=1, fit_intercept=False)

plt.figure()
plt.plot(-np.log10(alpha_array), coefs_enet[0,:,:].T)
plt.xlabel('-log10(alpha)')
plt.savefig('enet_path_coef_1.png')
plt.close()

plt.figure()
plt.plot(-np.log10(alpha_array), coefs_enet[1,:,:].T)
plt.xlabel('-log10(alpha)')
plt.savefig('enet_path_coef_2.png')
plt.close()


# Question: why there is possible for two solutions?

# Answer: because A is not full rank. so the null space is not zero.
#         but unfortunately, the sklearn.linear_models.ElasticNet didn't find the one with 
#         smallest norm coefficient. BUT! enet_path can find!!!



