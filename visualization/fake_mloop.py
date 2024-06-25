import mloop as ml
import numpy as np
import mloop.controllers as mlc
import mloop.interfaces as mli
import mloop.visualizations as mlv

def fake_experiment(pars):
    errs = np.random.normal(0,0.1)
    cost = -np.e**(-np.sum(pars)**2) + errs
    uncer = errs**2
    bad = False
    
    return {"cost": cost, "uncer": uncer, "bad": bad}

# Declare your custom class that inherits from the Interface class
class CustomInterface(mli.Interface):
    # Initialization of the interface, including this method is optional
    def __init__(self):
        # You must include the super command to call the parent class, Interface, constructor
        super(CustomInterface, self).__init__()

    def get_next_cost_dict(self, params_dict):
        # Get parameters from the provided dictionary
        params = params_dict["params"]
        cost_dict = fake_experiment(*params)

        return cost_dict


def main():

    params_labels = ["param_a", "param_b", "param_c", "param_d"]
    params_min_boundary = [0,0,0,0]
    params_max_boundary = [1,1,1,1]
    params_first_value = [0.9,0.9,0.9,0.9]
    max_iterations = 100
    target_cost = -1e9

    print(params_labels)
    # First create your interface
    interface = CustomInterface()

    # Next create the controller. Provide it with your interface and any options you want to set
    controller = mlc.create_controller(
        interface,
        max_num_runs=max_iterations,
        target_cost=target_cost,
        num_params=len(params_labels),
        min_boundary=params_min_boundary,
        max_boundary=params_max_boundary,
        params_names=params_labels,
        first_params=params_first_value,
    )
    # To run M-LOOP and find the optimal parameters just use the controller method optimize
    controller.optimize()

    # The results of the optimization will be saved to files and can also be accessed as attributes of the controller.
    print("Best parameters found:")
    print(controller.best_params)

    # You can also run the default sets of visualizations for the controller with one command
    mlv.show_all_default_visualizations(controller)

# Ensures main is run when this code is run as a script
if __name__ == "__main__":
    main()
