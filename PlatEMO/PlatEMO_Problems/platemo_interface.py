import matlab.engine
import numpy as np


def call_matlab_functions():
    # Start MATLAB engine
    eng = matlab.engine.start_matlab()
    # print(eng.path())
    eng.addpath(
        "CF",
        nargout=0,
    )

    # Create an instance of the CF1 class
    prob = eng.CF1()
    eng.Setting(prob)

    # Generate some random input for CalObj and CalCon methods
    X = matlab.double(np.random.rand(5, 10).tolist())  # Adjust the size of X as needed
    varargin = X

    pf = np.array(eng.GetOptimum(prob, 100))
    print(pf)

    # Call CalObj method
    pop_obj = eng.Evaluation(prob, varargin)

    breakpoint()

    # Display the result
    print("CalObj result:")
    print(np.array(pop_obj))

    # # Call CalCon method
    # pop_con = eng.CalCon(prob, X)

    # # Display the result
    # print("CalCon result:")
    # print(np.array(pop_con))

    # Stop the MATLAB engine
    eng.quit()


# Call the function
call_matlab_functions()
