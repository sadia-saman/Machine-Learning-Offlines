import numpy as np


def generateInvertibleSymmetric(n):
    mat = np.random.randint(0,100,(n,n))
    while np.linalg.det(mat)==0:
        mat = np.random.randint(0,100,(n,n))
    symm_mat = mat.dot(mat.T)

    return symm_mat
    

n=1

while(n<2):
    print("Enter the dimension of matrix (n>=2) : ")
    n = int(input())


symm_mat = generateInvertibleSymmetric(n)
print("original M ")
print(symm_mat)
print()
eigen_values, eigen_vectors = np.linalg.eig(symm_mat)
#print(eigen_values)

A = eigen_vectors
#print(eigen_vectors)
#print(A)

diag = np.zeros((n,n),float)
np.fill_diagonal(diag,eigen_values)

inv_A = np.linalg. inv(A) 
reconstructed_symm_mat = A.dot(diag.dot(inv_A))

print("reconstructed M ")
print(reconstructed_symm_mat)

print(np.allclose(symm_mat,reconstructed_symm_mat))