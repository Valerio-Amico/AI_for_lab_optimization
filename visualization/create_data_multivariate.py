import json
import numpy as np

if __name__ == "__main__":

    n_points = 100
    times = np.linspace(0,1,n_points)
    X=np.random.normal(size = n_points)
    Y=np.random.normal(size = n_points)
    Q=np.random.random(size = n_points)

    Z=np.exp(-X*X-Y*Y)*Q + np.random.random(size=n_points)*0.5

    dict = {
        "time" : list(times),
        "parameters" : {
            "par_x" : list(X),
            "par_y" : list(Y),
            "par_q" : list(Q)
        },
        "results" : {
            "cost" : list(Z),
            "uncer" : [0]*len(Z),
            "bad" : [True]*len(Z)
        }
    }

    with open("mydata_100.json","w") as file:
        json.dump(dict, file)