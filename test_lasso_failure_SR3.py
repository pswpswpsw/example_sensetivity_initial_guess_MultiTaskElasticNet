import numpy as np
from matplotlib import pyplot as plt

# TRUE MODEL -- noise less
np.random.seed(19901012)
B_L = np.array([[1, 0.1, 0.1 ,0.001 ,0.001,0.0001 ,0 ,0 ,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]).T
x = np.random.normal(0,0.3,(200,B_L.shape[0]))
# B = np.array([[1000, 0.1, -0.005, 0.2,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0]]).T
y = np.matmul(x, B_L)

print('std of flow states = ',np.std(y))

# y = y + np.random.normal(0,np.std(y)*0.01,(200,B_L.shape[1]))

print(B_L.shape)
print('std of flow states = ',np.std(y))

A = x
B = y

print('A = shape = ', A.shape)
print('B = shape = ', B.shape)

# sr3 solver - proximal operator of W21
def prox_w21(X, LAMBDA):
    TMP = np.diag([1 - LAMBDA/max(LAMBDA, np.linalg.norm(x)) for x in X.T])
    return np.matmul(X,TMP)

def prox_l1(X, LAMBDA):
    TMP = [np.sign(x) * max(abs(x) - LAMBDA,0) for x in X]
    return np.array(TMP)

X0 = np.linalg.lstsq(A,B)[0]

# print('OLS = ',X0)


# print(A)
# sr3 loop
def sr3(kappa, LAMBDA, A,B,C):
    HK = np.matmul(A.conjugate().T, A) + kappa * np.matmul(C.T, C)
    HK_inv = np.linalg.inv(HK)

    max_error = 1e-16
    i=0
    eta = 1./kappa

    # init W and Xk
    W = np.zeros(prox_l1(X=np.matmul(C, X0), LAMBDA=eta * LAMBDA).shape)
    tmp = np.matmul(A.conjugate().T, B) + kappa * np.matmul(C.T, W)
    Xk = np.matmul(HK_inv, tmp)

    rel_error = 1.0
    while rel_error > max_error:

        if i % 1600 == 0:
            Xk_last = Xk
        tmp = np.matmul(A.conjugate().T, B) + kappa * np.matmul(C.T,W)
        Xk = np.matmul(HK_inv, tmp)
        W = prox_w21(X=np.matmul(C, Xk), LAMBDA=eta * LAMBDA)

        # W = prox_l1(X=np.matmul(C,Xk), LAMBDA=eta*LAMBDA)
        i+=1

        if i % 1600 == 0:
            rel_error = np.min(abs(Xk_last - Xk)/abs(Xk))

    print('number of iter = ', i)
    # print(abs(Xk_last - Xk)/abs(Xk))

    return Xk


# start testing
num_alpha  = 30
kappa = 1
alpha_array = np.logspace(-8,3,num_alpha)
alpha_array = alpha_array[::-1]

coef_list = []
for index, alpha in enumerate(alpha_array):
    # print(alpha)
    coef_list.append(sr3(kappa=kappa, LAMBDA=alpha, A=A, B=B, C=np.eye(A.shape[1])))

coef_array = np.array(coef_list)

print(coef_array.shape)

plt.figure()
plt.ylabel('coef')
# plt.title('SR3-W21 single target: first one')
plt.semilogy(-np.log10(alpha_array), coef_array[:,:,0],'-o')
plt.ylim([1e-3, 1])
# plt.plot(-np.log10(alpha_array), coef_array[:,:])



plt.xlabel('-log10(alpha)')
plt.savefig('sr3_coef_1_my_toy_problem.png')
plt.close()

print(coef_array[-1,:,0])
print(coef_array[15,:,0])

print(coef_array[0,:,0])
# print("final result from sr3 = ", coef_array[-1,:,:])