import matlab.engine
import numpy as np


def call_matlab_functions():
    # Start MATLAB engine
    eng = matlab.engine.start_matlab()
    eng.addpath(
        eng.genpath("~/Documents/Richard/rrut_thesis_code/PlatEMO/PlatEMO_Problems"),
        nargout=0,
    )

    # Create an instance of the CF1 class
    prob = eng.CF1("D", 5)

    # Generate some random input for CalObj and CalCon methods
    X = matlab.double(np.random.rand(5, 5).tolist())  # Adjust the size of X as needed

    # Call CalObj method
    pop_obj = eng.Evaluation(prob, X)

    # Display the result
    print("CalObj result:")
    print(np.array(eng.getfield(pop_obj, "objs")))

    # Display the result
    print("CalCon result:")
    print(np.array(eng.getfield(pop_obj, "cons")))

    print("PF")
    print(np.array(eng.GetOptimum(prob, 100)))

    print("Bounds")
    print(np.array(eng.getfield(prob, "lower")))

    # Stop the MATLAB engine
    eng.quit()


# Call the function
call_matlab_functions()
