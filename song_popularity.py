'''
This is my first ML project . This is a linear regression model trained using the kaggle spotify dataset . 
It predicts the popularity of a song between 0-100 , based on the 7 input features 

'''

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import copy 

#takes only these features from the csv file  ,df = dataframe
df = pd.read_csv('SpotifyFeatures.csv', usecols=[
    'danceability', 'energy', 'loudness', 'tempo',
    'duration_ms', 'valence', 'popularity'])

#removes any NaN values 
df.dropna(inplace=True) 

# saves to a new file by the given name and prevents pandas from creating index column 
df.to_csv('spotify_clean.csv', index=False)

''' 
loading the data 
'''
data = np.genfromtxt('spotify_clean.csv', delimiter=',', skip_header=1)
#print(data.shape)
#print(data[:5]) 

X=data[:,1:]
y=data[:,0]
y=y.reshape(-1,1) 

'''
normalizing data 
'''

X_mean  = np.mean(X,axis=0)
X_std   = np.std(X,axis=0)                ### axis =0 means columns 

X= (X-X_mean)/X_std 

''' intializing parameters'''
m = X.shape[0]                          ### number of songs
n=X.shape[1]                            ### number of features

X = np.hstack((np.ones((m, 1)), X))  # add bias column as first column 
w = np.zeros((7,1))                  ### this includes the bias term 

alpha = 1e-03   
num_iter = 1e+05


### computes cost 
def compute_cost(X,y,m,w):
    prediction = X @ w
    error = prediction - y 
    cost = np.sum(error**2) / ( 2*m) 
    return cost 

### computes gradient descent 
def  compute_grad (X,y,w,alpha,num_iter):

    m = X.shape[0]
    n=X.shape[1] 
    cost_history =[]


    for i in range(num_iter):
        prediction = X @ w
        error = prediction - y 
        gradient = (X.T @ error) / m
        w = w - (alpha * gradient)

        
        if i % 10000 == 0:  
            cost = np.sum(error ** 2) / (2 * m)
            cost_history.append(cost)


    return w,cost_history


### final parameter value 
w_final, cost_history = compute_grad(X, y, w, alpha, int(num_iter))

final_cost = compute_cost(X, y, m, w_final)
print("Final Cost:", final_cost)



### plotting cost vs number of iterations 
plt.plot(range(0, len(cost_history)*10000, 10000), cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Reduction Over Time")
plt.grid(True)
plt.show()


# Predict on entire dataset
y_pred = X @ w_final

# Plot predicted vs actual
plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred, alpha=0.4, color='dodgerblue', label='Predictions')
plt.plot([0, 100], [0, 100], 'r--', label='Perfect prediction')  # reference line

plt.xlabel('Actual Popularity')
plt.ylabel('Predicted Popularity')
plt.title('Predicted vs Actual Popularity (Training Set)')
plt.legend()
plt.grid(True)
plt.show()






'''
predicting a songs popularity from its features' values
'''
### predicting any number of songs popularity 

while True:
    print("\nEnter song features (Ctrl+C to stop):")
    try:
        danceability = float(input("Danceability (0-1): "))
        energy       = float(input("Energy (0-1): "))
        loudness     = float(input("Loudness (in dB, e.g., -5): "))
        tempo        = float(input("Tempo (BPM): "))
        duration_ms  = float(input("Duration in ms: "))
        valence      = float(input("Valence (0-1): "))

        x_user = np.array([[danceability, energy, loudness, tempo, duration_ms, valence]])
        x_user = (x_user - X_mean) / X_std
        x_user = np.hstack((np.ones((1, 1)), x_user))

        predicted_popularity = x_user @ w_final
        predicted_popularity = np.clip(predicted_popularity, 0, 100)

        print(f"🎵 Predicted Popularity: {predicted_popularity[0][0]:.2f}")

    except KeyboardInterrupt:
        print("\nExiting prediction mode.")
        break








