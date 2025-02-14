import numpy as np

# Redéfinir D et Z
D = np.array([[1, 1, 0],
              [0, 1, 0],
              [0, 0, 1]])

Z = np.array([0, 1, 0, -1])

# Appliquer np.convolve sur les lignes de D et Z (mode='full' pour obtenir la convolution complète)
conv_results = []
for i in range(D.shape[0]):
    print("D : ", D.T[i])
    conv = np.convolve(D.T[i], Z, mode='full')
    print("D*Z : ",conv)
    conv_results.append(conv)

# Empiler les résultats colonne par colonne
y = np.column_stack(conv_results)

print("convolutions y = ")
print(y)

A = y@np.linalg.inv(D)
#A = np.array(([0,0,0],[1,0,0],[0,1,0],[-1,0,1],[0,-1,0],[0,0,-1]))

print("A : ")
print(A)

print("A.T : ")
print(A.T)

# Appliquer np.correlate sur les lignes de D et Z
correlation_results = []
for i in range(D.shape[0]):
    print("y : ", y[:,i])
    corr = np.correlate(y[:,i],Z, mode='valid')
    print("Z*y : ",corr)
    correlation_results.append(corr)

correlation_results = np.column_stack(correlation_results)

print("A^T @ y")
print(A.T @ y)

print("correlations : ")
print(correlation_results)

print("A@D")
print(A@D)