import json
import numpy as np

if __name__ == "__main__":

    n_points = 500
    X=np.random.random(n_points)
    Y=np.random.random(n_points)
    Q=np.random.random(n_points)

    Z=X*Y

    dict = {
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

    ciao = 5

    "ciao"

    with open("mydata.json","w") as file:
        json.dump(dict, file)