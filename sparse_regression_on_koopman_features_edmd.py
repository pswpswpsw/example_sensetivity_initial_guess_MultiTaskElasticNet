
# coding: utf-8

# In[1]:


import pickle
get_ipython().magic(u'pylab')
get_ipython().magic(u'matplotlib inline')


# ## Prepare data

# In[4]:


with open('top_i_selected_list_and_true_tj.pkl', 'rb') as f:
    top_i_selected_eigenTj_dict = pickle.load(f)

phi_tilde = top_i_selected_eigenTj_dict['phi_tilde'][-1]
X = top_i_selected_eigenTj_dict['X']


# In[6]:


print phi_tilde.shape
print X.shape


# ## Transform data into real type

# In[8]:


X_aug = np.vstack((X,np.zeros(X.shape)))

pr = np.real(phi_tilde)
pi = np.imag(phi_tilde)

phi_aug = np.vstack((np.hstack((pr, -pi)),np.hstack((pi, pr))))

# # center and standaradize
X_aug -= X_aug.mean(axis=0)
X_aug /= X_aug.std(axis=0)

phi_aug -= phi_aug.mean(axis=0)
phi_aug /= phi_aug.std(axis=0)


# In[9]:


print phi_aug.shape
print X_aug.shape


# In[10]:


print X_aug.mean(axis=0), X_aug.std(axis=0)
print phi_aug.mean(axis=0), phi_aug.std(axis=0)


# # running elastnet, multitaskElasticNet, not enet_path

# In[11]:


from sklearn.linear_model import ElasticNet, enet_path, lasso_path, MultiTaskElasticNet, Lasso

alpha_array = np.logspace(-16,1,200)
coef_list_1 = []
coef_list_2 = []
coef_list_1_m = [] 
coef_list_2_m = []
coef_list_1_l = [] 
coef_list_2_l = []

residual_list = []
residual_list_m = []
residual_list_l = []

for alpha in alpha_array:
    
    clf_l = Lasso(alpha=alpha, tol=1e-10, max_iter=3000, 
                     warm_start=False, random_state=1, fit_intercept=False)
    
    clf = ElasticNet(alpha=alpha, l1_ratio=0.5, tol=1e-10, max_iter=3000, 
                     warm_start=False, random_state=1, fit_intercept=False)
    
    clf_m = MultiTaskElasticNet(alpha=alpha, l1_ratio=0.5, tol=1e-10, max_iter=3000, 
                                warm_start=False, random_state=1, fit_intercept=False)
    # get fitted
    clf_m.fit(phi_aug, X_aug)
    clf.fit(phi_aug, X_aug)
    clf_l.fit(phi_aug, X_aug)
    
    # get predicted
    residual_list_m.append(np.linalg.norm(X_aug - clf_m.predict(phi_aug))**2/X_aug.shape[0])
    residual_list.append(  np.linalg.norm(X_aug - clf.predict(phi_aug))**2/X_aug.shape[0])
    residual_list_l.append(np.linalg.norm(X_aug - clf_l.predict(phi_aug))**2/X_aug.shape[0])
    
    coef_list_1.append(clf.coef_[0])
    coef_list_2.append(clf.coef_[1])
    coef_list_1_m.append(clf_m.coef_[0])
    coef_list_2_m.append(clf_m.coef_[1])
    coef_list_1_l.append(clf_l.coef_[0])
    coef_list_2_l.append(clf_l.coef_[1])


# In[50]:


print np.dot(phi_aug, coef_list_1[0])
print X_aug


# In[12]:


plt.plot(-np.log10(alpha_array), coef_list_1)
plt.xlabel(r'-$\log_{10}$($\alpha$)')


# In[13]:


plt.plot(-np.log10(alpha_array), coef_list_2)
plt.xlabel(r'-$\log_{10}$($\alpha$)')


# In[14]:


plt.plot(-np.log10(alpha_array), coef_list_1_m)
plt.xlabel(r'-$\log_{10}$($\alpha$)')


# In[15]:


plt.plot(-np.log10(alpha_array), coef_list_2_m)
plt.xlabel(r'-$\log_{10}$($\alpha$)')


# In[16]:


plt.plot(-np.log10(alpha_array), coef_list_1_l)
plt.xlabel(r'-$\log_{10}$($\alpha$)')


# In[17]:


plt.plot(-np.log10(alpha_array), coef_list_2_l)
plt.xlabel(r'-$\log_{10}$($\alpha$)')


# In[18]:



plt.semilogy(-np.log10(alpha_array), residual_list)
plt.xlabel(r'-$\log_{10}$($\alpha$)')
plt.ylabel('MSE')


# In[19]:



plt.semilogy(-np.log10(alpha_array), residual_list_l)
plt.xlabel(r'-$\log_{10}$($\alpha$)')
plt.ylabel('MSE')


# In[20]:



plt.semilogy(-np.log10(alpha_array), residual_list_m)
plt.xlabel(r'-$\log_{10}$($\alpha$)')
plt.ylabel('MSE')


# # Run elastic-net path

# In[21]:


get_ipython().magic(u'pinfo enet_path')


# In[22]:


alphas_enet, coefs_enet, _ = enet_path(phi_aug, X_aug, max_iter=3000, tol=1e-10, 
                                       alphas=np.logspace(-16,1,200),precompute=False,
                                       l1_ratio=0.5, random_state=1, fit_intercept=False)


# In[23]:


phi_aug.shape


# In[24]:


X_aug.shape


# In[25]:


coefs_enet.shape


# ## check the parameters

# ### First component

# In[26]:


for index,alpha in enumerate(alphas_enet):
    print '----'
    print 'alpha = ', alpha
    print coefs_enet[0,:,index].T


# ### Second component

# In[27]:


for index,alpha in enumerate(alphas_enet):
    print '----'
    print 'alpha = ', alpha
    print coefs_enet[1,:,index].T


# ## for each alpha, compute the corresponding residuals

# In[28]:


residual_list = []
for index,alpha in enumerate(alphas_enet):
    residual = X_aug - np.dot(phi_aug,coefs_enet[:,:,index].T)
    residual_list.append(np.linalg.norm(residual)**2/residual.shape[0])


# In[29]:


print X_aug.mean(axis=0)
print X_aug.std(axis=0)


# In[30]:


print phi_aug.mean(axis=0)
print phi_aug.std(axis=0)


# ## result from single enet

# In[65]:


np.vstack((coef_list_1[0], coef_list_2[0]))


# In[78]:


coef = np.vstack((coef_list_1[0], coef_list_2[0]))


# In[79]:


print np.dot(phi_aug, coef.T)
print X_aug


# In[71]:


phi_aug[]


# ## result from enet_path

# In[75]:


coefs_enet[:,:,-1]


# In[54]:


print np.dot(phi_aug, coefs_enet[:,:,-1].T)
print X_aug


# ## compare, so this is the reason!

# In[83]:


np.linalg.norm(np.dot(phi_aug, coef.T - coefs_enet[:,:,-1].T))


# ## check l1 norm differencen

# In[141]:


print 'l1 norm of enet path ', np.linalg.norm(coefs_enet[:,:,-1].T[:,0],1)
print 'l1 norm of single run ', np.linalg.norm(coef.T[:,0],1)


# ## check rank, it is certainly not full rnak

# In[113]:


print np.linalg.matrix_rank(phi_aug)
print phi_aug.shape[1]


# In[31]:


plt.plot(-np.log10(alphas_enet), coefs_enet[0,:,:].T)
plt.xlabel(r'-$\log_{10}$($\alpha$)')


# In[32]:


plt.plot(-np.log10(alphas_enet), coefs_enet[1,:,:].T)
plt.xlabel(r'-$\log_{10}$($\alpha$)')
# plt.ylim([-2,2])


# In[33]:


plt.semilogy(-np.log10(alphas_enet), residual_list)
plt.xlabel(r'-$\log_{10}$($\alpha$)')
plt.ylabel('MSE')


# In[34]:


from itertools import cycle


plt.figure(figsize=(16,8))
colors = cycle(['b', 'r', 'g', 'c', 'k'])
neg_log_alphas_lasso = -np.log10(alphas_lasso)
neg_log_alphas_enet = -np.log10(alphas_enet)

for coef_l, coef_e, c in zip(coefs_lasso, coefs_enet, colors):
    l1 = plt.plot(neg_log_alphas_lasso, coef_l.T, c=c)
    l2 = plt.plot(neg_log_alphas_enet, coef_e.T, linestyle='--', c=c)

plt.xlabel('-Log(alpha)')
plt.ylabel('coefficients')
plt.title('Lasso and Elastic-Net Paths')
plt.legend((l1[-1], l2[-1]), ('Lasso', 'Elastic-Net'), loc='lower left')
plt.axis('tight')


# ## Can we add CV on the above?

# In[122]:


# result are too bad!

from sklearn.linear_model import ElasticNetCV, MultiTaskElasticNetCV

coef_list_enet_cv_1 = []
coef_list_enet_cv_2 = []



clf_enet_cv = MultiTaskElasticNetCV(alphas=alpha_array, tol=1e-10, max_iter=3000, cv=5, random_state=1, fit_intercept=False)

# get fitted
clf_enet_cv.fit(phi_aug, X_aug)






# ## 1-norm

# In[144]:


clf_enet_cv.coef_


# ## so the norm for CV implemented is still not optimizing the l1 coefficients

# In[145]:


np.linalg.norm(clf_enet_cv.coef_[0,:],1)


# # How about multiTask?

# In[ ]:


# performance is bad


# # How about using glmnet?

# In[1]:


from glmnet import glmnet

