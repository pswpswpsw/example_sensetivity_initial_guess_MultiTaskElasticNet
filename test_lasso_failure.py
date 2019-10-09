import numpy as np
from matplotlib import pyplot as plt
plt.style.use('siads')
from sklearn.linear_model import lasso_path

# TRUE MODEL -- noise less
np.random.seed(19901012)
B_L = np.array([[1, 0.1, 0.1 ,0.001 ,0.001,0.0001 ,0 ,0 ,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]).T
x = np.random.normal(0,0.3,(200,B_L.shape[0]))
# B = np.array([[1000, 0.1, -0.005, 0.2,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0]]).T
y = np.matmul(x, B_L)

print('std of flow states = ',np.std(y))

# y = y + np.random.normal(0,np.std(y)*0.05,(200,B_L.shape[1]))

print(B_L.shape)


print('std of features', np.std(x,axis=0))

log_alpha = np.linspace(-16,-2.5,50)


C = lasso_path(x,y, alphas=10**(log_alpha),eps=1e-10)

plt.loglog(C[0], abs(C[1][0,:,:].T), '-^')
plt.grid()

# plt.loglog(C[0], abs(B[2]*np.ones(C[0].shape[0])),'g--')
# plt.loglog(C[0], abs(B[3]*np.ones(C[0].shape[0])),'b--')
plt.loglog(C[0], 0.01*np.min(abs(C[1][0,1:6,:].T),axis=1),'k--')



plt.ylim([1e-8, 1e-1])
plt.show()
plt.close()


# residuals




print(C[0][-1], ' final coefficients = ', C[1][0,:,-1])
