import numpy as np


#theta : parameters
#X : matrix of all instances and features
#y: vectors of all the results

def sigmoid(z):
    return 1/(1+np.exp(-z))

def calculate_gradient(theta, X, y):
    m = y.size
    h = sigmoid(X.dot(theta))
    gradient = (1/m) * X.T.dot(h-y)
    return gradient

def gradient_descent(X, y, alpha=0.1, num_iter=1000, tol=1e-7):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    theta = np.zeros(X_b.shape[1])


    for i in range(num_iter):
        grad = calculate_gradient(theta, X_b, y)
        theta -= alpha*grad

        if np.linalg.norm(grad) < tol:
            print(f"Convergence reached at iteration {i}")
            break
    
    return theta

def predict_prob(X, theta):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    print(X_b)
    return sigmoid(X_b @ theta)

def predict(X, theta, threshold=0.5):
    return (predict_prob(X, theta) >= threshold).astype(int)

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

theta = gradient_descent(X_train_scaled, y_train, alpha=0.1)

y_pred_train = predict(X_train_scaled, theta)
y_pred_test = predict(X_test_scaled, theta)

accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test) 

print(f"Training Accuracy: {accuracy_train}")
print(f"Testing Accuracy: {accuracy_test}")