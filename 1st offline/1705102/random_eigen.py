import numpy as np

def generateInvertible_M(n):
    mat = np.random.randint(0,100,(n,n))  
    while np.linalg.det(mat)==0:
        mat = np.random.randint(0,100,(n,n))

    return mat

n=1
while(n<2):
    print("Enter the dimension of matrix (n>=2) : ")
    n = int(input())

mat = generateInvertible_M(n)

print("original M ")
print(mat)
print()
eigen_values, eigen_vectors = np.linalg.eig(mat)
#print(eigen_values)

A = eigen_vectors
#print(eigen_vectors)
#print(A)

diag = np.zeros((n,n),float)
np.fill_diagonal(diag,eigen_values)

inv_A = np.linalg. inv(A) 
reconstructed_M = A.dot(diag.dot(inv_A))

print("reconstructed M ")
print(reconstructed_M)

print(np.allclose(mat,reconstructed_M))