import numpy as np

print("Enter the dimension of matrix (n,m)")
n = int(input())
m = int(input())

mat = np.random.randint(0,100,(n,m))
u, s, v = np.linalg.svd(mat, full_matrices=True)

pseudo_inv_m = np.linalg.pinv(mat, rcond=1e-15, hermitian=False)


print('pseudo inverse of mat')
print(pseudo_inv_m)




## equation of Moore-Penrose Pseudoinverse (A+) = V.(D+).(U.T)
D =  np.zeros((len(v[0]),len(u[0])))
for i in range(len(s)): 
    if s[i]!=0 : 
        D[i][i]=(1/s[i])  

###    Dplus = D.T
Dplus = D
print("shape")
print(np.shape(v))
print(np.shape(Dplus))
print(np.shape(u))


pseudo_inv_m2 = np.dot(v.T,np.dot(Dplus,u.T))

print('pseudo inverse of mat (manual)')
print(pseudo_inv_m2)

print(np.allclose(pseudo_inv_m,pseudo_inv_m2))