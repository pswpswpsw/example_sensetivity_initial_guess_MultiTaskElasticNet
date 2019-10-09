import numpy as np
from matplotlib import pyplot as plt

# prepare data
data = np.load('data.npz')
A = data['A']
B = data['B']
print(A.shape) # (1600,14)
print(B.shape) # (1600,2)

# sr3 solver - proximal operator of W21

def prox_w21(X, LAMBDA):
    TMP = np.diag([1 - LAMBDA/max(LAMBDA, np.linalg.norm(x)) for x in X.T])
    return np.matmul(X,TMP)

def prox_l1(X, LAMBDA):
    TMP = [np.sign(x) * max(abs(x) - LAMBDA,0) for x in X]
    return np.array(TMP)

X0 = np.linalg.lstsq(A,B)[0]

# print(A)
# sr3 loop
def sr3(kappa, LAMBDA, A,B,C):
    HK = np.matmul(A.conjugate().T, A) + kappa * np.matmul(C.T, C)
    HK_inv = np.linalg.inv(HK)

    max_error = 1e-16
    i=0
    eta = 1./kappa

    # init W and Xk
    W = np.zeros(prox_w21(X=np.matmul(C, X0), LAMBDA=eta * LAMBDA).shape)
    tmp = np.matmul(A.conjugate().T, B) + kappa * np.matmul(C.T, W)
    Xk = np.matmul(HK_inv, tmp)

    rel_error = 1.0
    while rel_error > max_error:

        if i % 100 == 0:
            Xk_last = Xk
        tmp = np.matmul(A.conjugate().T, B) + kappa * np.matmul(C.T,W)
        Xk = np.matmul(HK_inv, tmp)
        W = prox_w21(X=np.matmul(C,Xk), LAMBDA=eta*LAMBDA)
        i+=1

        if i % 100 == 0:
            rel_error = np.min(abs(Xk_last - Xk)/abs(Xk))

    print('number of iter = ', i)
    # print(abs(Xk_last - Xk)/abs(Xk))

    return Xk


# start testing
num_alpha  = 30
kappa = 1
alpha_array = np.logspace(-16,16,num_alpha)
alpha_array = alpha_array[::-1]

coef_list = []
for index, alpha in enumerate(alpha_array):
    print(alpha)
    coef_list.append(sr3(kappa=kappa, LAMBDA=alpha, A=A, B=B, C=np.eye(A.shape[1])))

coef_array = np.array(coef_list)

print(coef_array.shape)

plt.figure()
plt.ylabel('coef')
# plt.title('SR3-W21 single target: first one')
plt.plot(-np.log10(alpha_array), coef_array[:,:,0])
# plt.plot(-np.log10(alpha_array), coef_array[:,:])

plt.xlabel('-log10(alpha)')
plt.savefig('sr3_coef_1.png')
plt.close()

plt.figure()
plt.ylabel('coef')
# plt.title('SR3-W21 single target: second one')
plt.plot(-np.log10(alpha_array), coef_array[:,:,1])
plt.xlabel('-log10(alpha)')
plt.savefig('sr3_coef_2.png')
plt.close()

print("final result from sr3 = ", coef_array[-1,:,:])