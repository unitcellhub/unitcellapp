from numpy import *

# For the custom equation feature in the Explore tab, it is possible the
# user would like to use math functions such as exp and sin. Normally,
# these are references against numpy as numpy.exp (or np.exp when numpy
# is imported as np). This is less than idea from a user experience
# perspective, so it make sense for numpy to be fully loaded into the
# name space for these custom evaluations. Unfortunately, wildcard
# imports are not supported within a function definition, so this custom
# submodule was define to circumvent this issu.


# Define an evaluation function for custom equations that full loads
# numpy
def customeval(data, eqn):
    """ Custom evaluation function 
    
    Arguments
    ---------
    data: dictionary or pandas DataFrame is reference data
    eqn: str defining the custom equation (and that references *data* as
         needed)
        
    
    Returns
    -------
    Float or 1D array with the same number of rows as *data*
    
    """
    return eval(eqn)