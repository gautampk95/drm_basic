#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# importing libraries needed
import scipy
from scipy.stats import norm

import numpy as np
import pandas as pd
# from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings


# In[ ]:


def drm_summary(model):
    """
    Summary function for dose-response model optimization results.

    This function provides a summary of the optimization results from a dose-response model, 
    displaying the optimized parameter values, the objective function value (Sum of Squared Errors), 
    and information regarding the success of the optimization process.

    Parameters:
    ----------
    model : tuple
        A tuple containing the optimization result and model information. The first element (`model[0]`) 
        is the optimization result object, which should include attributes like `x` (the optimized parameters), 
        `fun` (the objective function value, typically SSE), `success` (boolean indicating if the optimization succeeded), 
        and `message` (a message describing the result). The second element (`model[1]`) is a string describing 
        the model type/name.

    Returns:
    -------
    None
        This function prints the model summary, including:
        1. The model name or type.
        2. The optimized parameter values, if available.
        3. The value of the objective function (SSE).
        4. The success status of the optimization.
        5. Any message returned by the optimizer.
    
    Notes:
    ------
    - The parameter names used in the output are `b`, `c`, `d`, `e`, and `f`, and the corresponding parameter 
      values from the optimization are printed if the optimization was successful.
    - If the optimization fails, an error message is displayed, and the user is advised to try different 
      initial parameter values.
    """
    # estimated parameters' names
    param_names = model[3]

    # 
    if model[0].success:
        params = model[0].x
        print(f"Model Summary:\n")
        print(f"{model[1]}")
        formatted_params = []
        for i, param in enumerate(params):
            if i < len(param_names):  # Ensure we don't exceed the number of names
                formatted_params.append(f"{param_names[i]}={param}")
    
        # Join the parameters into a single string
        result = ", ".join(formatted_params)
        print(f"Optimized Parameters:\n {result}\n")
        
        # Additional information can be added
        print(f"Objective Function Value (SSE): {model[0].fun:.4f}")
        print(f"Optimization Success: {model[0].success}")
        print(f"Message: {model[0].message}")
    else:
        print(f"Optimization failed.")
        print(f"Message: {model[0].message}")
        print(f"---Try using different initial parameter values---")


# In[1]:


def drm_predict(model, x):
    """
    Predicts responses for dose-response models (DRM) based on the model type.

    This function predicts the response `y_pred` for a given explanatory variable `x` using different types of dose-response models, including Log-logistic models (2-5 parameter), Brain-Cousens models (4-5 parameter), Weibull models, Gompertz models, and the Log-normal model. The appropriate model is chosen based on the model type string in the `model` input.

    Parameters:
    ----------
    model : tuple
        A tuple containing:
        1. Optimization result from fitting the model (usually `scipy.optimize.OptimizeResult`).
        2. Additional information about the model fit (if any).
        3. A string representing the type of model, such as "LL2", "LL3", "LL4", "LL5", "BC4", "BC5", "Weib1", "Weib2", "Gomp4", or "log_normal".
    x : array-like
        The explanatory variable (dose) for which predictions are to be made.

    Returns:
    -------
    y_pred : array-like
        Predicted response values for the input dose `x`.

    Model Types:
    ------------
    - **LL2**: 2-parameter Log-logistic model.
    - **LL3**: 3-parameter Log-logistic model.
    - **LL4**: 4-parameter Log-logistic model.
    - **LL5**: 5-parameter Log-logistic model.
    - **BC4**: 4-parameter Brain-Cousens model.
    - **BC5**: 5-parameter Brain-Cousens model.
    - **Weib1**: Weibull Type 1 model.
    - **Weib2**: Weibull Type 2 model.
    - **Gomp4**: 4-parameter Gompertz model.
    - **log_normal**: Log-normal model.
    
    Raises:
    -------
    RuntimeError
        If the model type is not recognized or supported, an error is raised with a message prompting to check the code.
    
    Notes:
    ------
    - Each model follows its respective mathematical equation to predict the response `y_pred`.
    - The models are typically based on nonlinear regression with different numbers of parameters to describe dose-response relationships.
    """
    
    ## Fitting the model/predicting on test data

    # 2-parameter Log-logistic model
    if model[2] == "LL2":
        b, e = model[0].x
        y_pred = 1/(1 + np.exp(b*(np.log(x + 1e-10) - np.log(e))))

    # 3-parameter Log-logistic model
    elif model[2] == "LL3":
        b, d, e = model[0].x
        y_pred = d/(1 + np.exp(b*(np.log(x + 1e-10) - np.log(e))))

    # 4-parameter Log-logistic model
    elif model[2] == "LL4":
        b, c, d, e = model[0].x
        y_pred = c + ((d - c)/(1 + np.exp(b*(np.log(x + 1e-10) - np.log(e)))))

    # 5-parameter Log-logistic model
    elif model[2] == "LL5":
        b, c, d, e, f = model[0].x
        y_pred = c + ((d - c )/((1 + np.exp(b*(np.log(x + 1e-10) - np.log(e))))**f))

    # 5-parameter Brain-Cousens model
    elif model[2] == "BC5":
        b, c, d, e, f = model[0].x
        y_pred = c + ((d - c + f*x)/(1+np.exp(b*(np.log(x + 1e-10)-np.log(e)))))

    # 5-parameter Brain-Cousens model
    elif model[2] == "BC4":
        b, d, e, f = model[0].x
        y_pred = ((d + f*x)/(1 + np.exp(b*(np.log(x + 1e-10)-np.log(e)))))

    # Weibull Type 1 Model
    elif model[2] == "Weib1":
        b, c, d, e = model[0].x
        y_pred = c + ((d-c)*np.exp(-np.exp(b*(np.log(x + 1e-10)-np.log(e)))))

    # Weibull Type 2 Model
    elif model[2] == "Weib2":
        b, c, d, e = model[0].x
        y_pred = c + ((d-c)*(1-np.exp(-np.exp(b*(np.log(x + 1e-10)-np.log(e))))))
      
    # 4-parameter Gompertz model
    elif model[2] == "Gomp4":
        b, c, d, e = model[0].x
        y_pred = c + ((d-c)*(np.exp(-np.exp(b*(x-e)))))

    # Log-normal model
    elif model[2] == "log_normal":
        b, c, d, e = model[0].x
        y_pred = c + (d - c) * norm.cdf(b * (np.log(x + 1e-10) - np.log(e)))

    else:
        raise RuntimeError(f"--Something went wrong. Please check your code or the model you built cannot be fitted--")

    return y_pred
        

