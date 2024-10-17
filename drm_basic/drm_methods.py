#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# importing libraries needed
import scipy
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.stats import norm

import numpy as np
import pandas as pd
# from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings


# In[ ]:


def LL2(y, x, par_init = [1, 50], opt = "L-BFGS-B", fix_par = [None, None]):
    """
    2-parameter Log-logistic modeling function.

    This function models a response `y` as a function of dose `x` using a 2-parameter Log-logistic model, 
    where the parameters `c` and `d` are fixed at 0 and 1, respectively.

    Parameters:
    ----------
    y : array-like
        Response or target variable (dependent variable).
    x : array-like
        Dose or explanatory variable (independent variable).
    par_init : list, optional
        Initial values for the two parameters `b` and `e` to be estimated, in the form [b, e]. 
        Default is [1, 50].
    opt : str, optional
        Optimization method. Default is "L-BFGS-B", which uses bounds. Other methods such as 
        "Nelder-Mead", "TNC", "Powell", and "SLSQP" can also be used, though support for these methods may not be fully verified.
    fix_par : list, optional
        Fixed parameter values for `b` and `e`. If a parameter should be estimated, set it to None.
        Default is [None, None].

    Returns:
    -------
    tuple
        A tuple with four elements:
        1. Optimization result containing the estimated parameters and other details from the minimization of SSE.
        2. A string indicating the type/name of the model ("2-parameter Log-logistic model").
        3. The function name ("LL2"), useful for future reference.
        4. A list of estimated parameters: ["b", "e"].

    Raises:
    -------
    ValueError:
        If the number of initial parameters in `par_init` does not match the number of fixed parameters in `fix_par`.

    Notes:
    ------
    - In this model, the parameter `c` is fixed at 0 and `d` is fixed at 1.
    - The "L-BFGS-B" method is designed for optimization with bounds, and this function provides bounds for 
      the parameter `e` to ensure it is positive.
    """

    # Checking the number of fixed parameters passed
    if len(par_init) != len(fix_par):
        raise ValueError("Mention either 'None' or numerical values to b, e parameters (c is already fixed to zero and d is fixed to 1)")

    # L-BFGS-B can accept bounds. Also, few more optimization methods work well with bounds  
    if opt in ["L-BFGS-B", "Nelder-Mead", "TNC", "Powell", "SLSQP"]:
        bounds = [(None, None),  # No bound for b
              (1e-10, None)] # Bound e to be positive
    else:
        bounds = None
        
    # define objective function, c = 0 (fixed), d = 1 (fixed)
    def objective(par, y, x, fix):
        par = [fix[i] if fix[i] is not None else par[i] for i in range(len(par))]
        b, e = par
        y_ = 1/(1 + np.exp(b*(np.log(x+1e-10) - np.log(e))))
        return np.sum((y - y_)**2)
    
    res_ = minimize(objective, par_init, args = (y, x, fix_par), method=opt, bounds=bounds, options = {'maxiter' : 10000})

    # Replace any fixed parameters with their actual values in the result
    res_.x = [float(fix_par[i]) if fix_par[i] is not None else res_.x[i] for i in range(len(res_.x))]
    
    return (res_, "2-parameter Log-logistic model", "LL2", ["b", "e"])


# In[ ]:


def LL3(y, x, par_init = [1, 100, 50], opt = "L-BFGS-B", fix_par = [None, None, None]):
    """
    3-parameter Log-logistic modeling function.

    This function models a response `y` as a function of dose `x` using a 3-parameter Log-logistic model, 
    where the parameter `c` is fixed at 0.

    Parameters:
    ----------
    y : array-like
        Response or target variable (dependent variable).
    x : array-like
        Dose or explanatory variable (independent variable).
    par_init : list, optional
        Initial values for the three parameters `b`, `d`, and `e` to be estimated, in the form [b, d, e]. 
        Default is [1, 100, 50].
    opt : str, optional
        Optimization method. Default is "L-BFGS-B", which uses bounds. Other methods such as 
        "Nelder-Mead", "TNC", "Powell", and "SLSQP" can also be used, though support for these methods may not be fully verified.
    fix_par : list, optional
        Fixed parameter values for `b`, `d`, and `e`. If a parameter should be estimated, set it to None.
        Default is [None, None, None].

    Returns:
    -------
    tuple
        A tuple with four elements:
        1. Optimization result containing the estimated parameters and other details from the minimization of SSE.
        2. A string indicating the type/name of the model ("3-parameter Log-logistic model").
        3. The function name ("LL3"), useful for future reference.
        4. A list of estimated parameters: ["b", "d", "e"].

    Raises:
    -------
    ValueError:
        If the number of initial parameters in `par_init` does not match the number of fixed parameters in `fix_par`.

    Notes:
    ------
    - In this model, the parameter `c` is fixed at 0.
    - The "L-BFGS-B" method is designed for optimization with bounds, and this function provides bounds for 
      the parameter `e` to ensure it is positive.
    """

    # Checking the number of fixed parameters passed
    if len(par_init) != len(fix_par):
        raise ValueError("Mention either 'None' or numerical values to b, d, e parameters (c is already fixed to zero)")

    # L-BFGS-B can accept bounds. Also, few more optimization methods work well with bounds  
    if opt in ["L-BFGS-B", "Nelder-Mead", "TNC", "Powell", "SLSQP"]:
        bounds = [(None, None),  # No bound for b
              (None, None),  # No bound for d
              (1e-10, None)] # Bound e to be positive
    else:
        bounds = None
        
    # define objective function, c = 0 (fixed)
    def objective(par, y, x, fix):
        par = [fix[i] if fix[i] is not None else par[i] for i in range(len(par))]
        b, d, e = par
        y_ = d/(1 + np.exp(b*(np.log(x+1e-10) - np.log(e))))
        return np.sum((y - y_)**2)
    
    res_ = minimize(objective, par_init, args = (y, x, fix_par), method=opt, bounds=bounds, options = {'maxiter' : 10000})

    # Replace any fixed parameters with their actual values in the result
    res_.x = [float(fix_par[i]) if fix_par[i] is not None else res_.x[i] for i in range(len(res_.x))]
    
    return (res_, "3-parameter Log-logistic model", "LL3", ["b", "d", "e"])


# In[ ]:


def LL4(y, x, par_init = [1, 0, 100, 50], opt = "L-BFGS-B", fix_par = [None, None, None, None]):
    """
    4-parameter Log-logistic modeling function.

    This function models a response `y` as a function of dose `x` using a 4-parameter Log-logistic model.

    Parameters:
    ----------
    y : array-like
        Response or target variable (dependent variable).
    x : array-like
        Dose or explanatory variable (independent variable).
    par_init : list, optional
        Initial values for the four parameters `b`, `c`, `d`, and `e` to be estimated, in the form [b, c, d, e]. 
        Default is [1, 0, 100, 50].
    opt : str, optional
        Optimization method. Default is "L-BFGS-B", which utilizes bounds. Other methods such as 
        "Nelder-Mead", "TNC", "Powell", and "SLSQP" can also be used, though support for these methods may not be fully verified.
    fix_par : list, optional
        Fixed parameter values for `b`, `c`, `d`, and `e`. If a parameter should be estimated, set it to None.
        Default is [None, None, None, None].

    Returns:
    -------
    tuple
        A tuple with four elements:
        1. Optimization result containing the estimated parameters and other details from the minimization of SSE.
        2. A string indicating the type/name of the model ("4-parameter Log-logistic model").
        3. The function name ("LL4"), useful for future reference.
        4. A list of estimated parameters: ["b", "c", "d", "e"].

    Raises:
    -------
    ValueError:
        If the number of initial parameters in `par_init` does not match the number of fixed parameters in `fix_par`.

    Notes:
    ------
    - The "L-BFGS-B" method is designed for optimization with bounds, and this function provides bounds for 
      the parameter `e` to ensure it is positive.
    """

    # Checking the number of fixed parameters passed
    if len(par_init) != len(fix_par):
        raise ValueError("Mention either 'None' or numerical values to b, c, d, e parameters.")

    # L-BFGS-B can accept bounds. Also, few more optimization methods work well with bounds  
    if opt in ["L-BFGS-B", "Nelder-Mead", "TNC", "Powell", "SLSQP"]:
        bounds = [(None, None),  # No bound for b
              (None, None),  # No bound for c
              (None, None),  # No bound for d
              (1e-10, None)] # Bound e to be positive
    else:
        bounds = None
        
    # define objective function
    def objective(par, y, x, fix):
        par = [fix[i] if fix[i] is not None else par[i] for i in range(len(par))]
        b, c, d, e = par
        y_ = c + ((d - c)/(1 + np.exp(b*(np.log(x+1e-10) - np.log(e)))))
        return np.sum((y - y_)**2)
    
    res_ = minimize(objective, par_init, args = (y, x, fix_par), method=opt, bounds=bounds, options = {'maxiter' : 10000})

    # Replace any fixed parameters with their actual values in the result
    res_.x = [float(fix_par[i]) if fix_par[i] is not None else res_.x[i] for i in range(len(res_.x))]
    
    return (res_, "4-parameter Log-logistic model", "LL4", ["b", "c", "d", "e"])


# In[ ]:


def LL5(y, x, par_init = [1, 0, 100, 50, 1], opt = "L-BFGS-B", fix_par = [None, None, None, None, None]):
    """
    5-parameter Log-logistic modeling function.

    This function models a response `y` as a function of dose `x` using a 5-parameter Log-logistic model.

    Parameters:
    ----------
    y : array-like
        Response or target variable (dependent variable).
    x : array-like
        Dose or explanatory variable (independent variable).
    par_init : list, optional
        Initial values for the five parameters `b`, `c`, `d`, `e`, and `f` to be estimated, in the form [b, c, d, e, f]. 
        Default is [1, 0, 100, 50, 1].
    opt : str, optional
        Optimization method. Default is "L-BFGS-B", which utilizes bounds. Other methods such as 
        "Nelder-Mead", "TNC", "Powell", and "SLSQP" can also be used, though support for these methods may not be fully verified.
    fix_par : list, optional
        Fixed parameter values for `b`, `c`, `d`, `e`, and `f`. If a parameter should be estimated, set it to None.
        Default is [None, None, None, None, None].

    Returns:
    -------
    tuple
        A tuple with four elements:
        1. Optimization result containing the estimated parameters and other details from the minimization of SSE.
        2. A string indicating the type/name of the model ("5-parameter Log-logistic model").
        3. The function name ("LL5"), useful for future reference.
        4. A list of estimated parameters: ["b", "c", "d", "e", "f"].

    Raises:
    -------
    ValueError:
        If the number of initial parameters in `par_init` does not match the number of fixed parameters in `fix_par`.

    Notes:
    ------
    - The "L-BFGS-B" method is designed for optimization with bounds, and this function provides bounds for 
      the parameters `e` and `f` to ensure they are positive.
    """

    # Checking the number of fixed parameters passed
    if len(par_init) != len(fix_par):
        raise ValueError("Mention either 'None' or numerical values to b, c, d, e, f parameters.")

    # L-BFGS-B can accept bounds. Also, few more optimization methods work well with bounds  
    if opt in ["L-BFGS-B", "Nelder-Mead", "TNC", "Powell", "SLSQP"]:
        bounds = [(None, None),  # No bound for b
                  (None, None),  # No bound for c
                  (None, None),  # No bound for d
                  (1e-10, None), # Bound e to be positive
                  (1e-10, None)] # Bound f to be positive
    else:
        bounds = None
    
    # define objective function
    def objective(par, y, x, fix):
        par = [fix[i] if fix[i] is not None else par[i] for i in range(len(par))]
        b, c, d, e, f = par
        y_ = c + ((d - c )/((1 + np.exp(b*(np.log(x) - np.log(e))))**f))
        return np.sum((y - y_)**2)
    
    res_ = minimize(objective, par_init, args = (y, x, fix_par), method=opt, bounds=bounds, options = {'maxiter' : 10000})

    # Replace any fixed parameters with their actual values in the result
    res_.x = [float(fix_par[i]) if fix_par[i] is not None else res_.x[i] for i in range(len(res_.x))]
    
    return (res_, "5-parameter Log-logistic model", "LL5", ["b", "c", "d", "e", "f"])


# In[ ]:


def BC5(y, x, par_init = [1, 0, 100, 50, 1], opt = "L-BFGS-B", fix_par = [None, None, None, None, None]):
    """
    5-parameter Brain-Cousens modeling function.

    This function models a response `y` as a function of dose `x` using a 5-parameter Brain-Cousens model.

    Parameters:
    ----------
    y : array-like
        Response or target variable (dependent variable).
    x : array-like
        Dose or explanatory variable (independent variable).
    par_init : list, optional
        Initial values for the five parameters `b`, `c`, `d`, `e`, and `f` to be estimated, in the form [b, c, d, e, f]. 
        Default is [1, 0, 100, 50, 1].
    opt : str, optional
        Optimization method. Default is "L-BFGS-B", which utilizes bounds. Other methods such as 
        "Nelder-Mead", "TNC", "Powell", and "SLSQP" can also be used, though support for these methods may not be fully verified.
    fix_par : list, optional
        Fixed parameter values for `b`, `c`, `d`, `e`, and `f`. If a parameter should be estimated, set it to None.
        Default is [None, None, None, None, None].

    Returns:
    -------
    tuple
        A tuple with four elements:
        1. Optimization result containing the estimated parameters and other details from the minimization of SSE.
        2. A string indicating the type/name of the model ("5-parameter Brain-Cousens model").
        3. The function name ("BC5"), useful for future reference.
        4. A list of estimated parameters: ["b", "c", "d", "e", "f"].

    Raises:
    -------
    ValueError:
        If the number of initial parameters in `par_init` does not match the number of fixed parameters in `fix_par`.

    Notes:
    ------
    - The "L-BFGS-B" method is designed for optimization with bounds, and this function provides bounds for 
      the parameter `e` to ensure it is positive, while `f` has no specific bounds.
    """

    # Checking the number of fixed parameters passed
    if len(par_init) != len(fix_par):
        raise ValueError("Mention either 'None' or numerical values to b, c, d, e, f parameters.")

    # L-BFGS-B can accept bounds. Also, few more optimization methods work well with bounds   
    if opt in ["L-BFGS-B", "Nelder-Mead", "TNC", "Powell", "SLSQP"]:
        bounds = [(None, None),  # No bound for b
                  (None, None),  # No bound for c
                  (None, None),  # No bound for d
                  (1e-10, None), # Bound e to be positive
                  (None, None)]  # No bound for f
    else:
        bounds = None
        
    # define objective function
    def objective(par, y, x, fix):
        par = [fix[i] if fix[i] is not None else par[i] for i in range(len(par))]
        b, c, d, e, f = par
        y_ = c + ((d - c + f*x)/(1 + np.exp(b*(np.log(x + 1e-10)-np.log(e)))))
        return np.sum((y - y_)**2)

    res_ = minimize(objective, par_init, args = (y, x, fix_par), method=opt, bounds=bounds, options = {'maxiter' : 10000})

    # Replace any fixed parameters with their actual values in the result
    res_.x = [float(fix_par[i]) if fix_par[i] is not None else res_.x[i] for i in range(len(res_.x))]
    
    return (res_, "5-parameter Brain-Cousens model", "BC5", ["b", "c", "d", "e", "f"])


# In[ ]:


def BC4(y, x, par_init=[1, 1, 50, 1], opt="L-BFGS-B", fix_par=[None, None, None, None]):
    """
    4-parameter Brain-Cousens modeling function.

    This function models a response `y` as a function of dose `x` using a 4-parameter Brain-Cousens model, which is derived when the parameter `c` is fixed to zero in the 5-parameter Brain-Cousens model.

    Parameters:
    ----------
    y : array-like
        Response or target variable (dependent variable).
    x : array-like
        Dose or explanatory variable (independent variable).
    par_init : list, optional
        Initial values for the four parameters `b`, `d`, `e`, and `f` to be estimated, in the form [b, d, e, f]. 
        Default is [1, 1, 50, 1].
    opt : str, optional
        Optimization method. Default is "L-BFGS-B", which utilizes bounds. Other methods such as 
        "Nelder-Mead", "TNC", "Powell", and "SLSQP" can also be used, though support for these methods may not be fully verified.
    fix_par : list, optional
        Fixed parameter values for `b`, `d`, `e`, and `f`. If a parameter should be estimated, set it to None.
        Default is [None, None, None, None]. Note that `c` is fixed at zero.

    Returns:
    -------
    tuple
        A tuple with four elements:
        1. Optimization result containing the estimated parameters and other details from the minimization of SSE.
        2. A string indicating the type/name of the model ("4-parameter Brain-Cousens model").
        3. The function name ("BC4"), useful for future reference.
        4. A list of estimated parameters: ["b", "d", "e", "f"].

    Raises:
    -------
    ValueError:
        If the number of initial parameters in `par_init` does not match the number of fixed parameters in `fix_par`.

    Notes:
    ------
    - The "L-BFGS-B" method is designed for optimization with bounds. This function provides bounds for 
      the parameter `e` to ensure it is positive, while `f` has no specific bounds.
    """
    
    # Checking the number of fixed parameters passed
    if len(par_init) != len(fix_par):
        raise ValueError("Mention either 'None' or numerical values to b, d, e, f parameters.")

    # L-BFGS-B can accept bounds. Also, few more optimization methods work well with bounds   
    if opt in ["L-BFGS-B", "Nelder-Mead", "TNC", "Powell", "SLSQP"]:
        bounds = [(None, None),  # No bound for b
                  (None, None),  # No bound for d
                  (1e-10, None), # Bound e to be positive
                  (None, None)]  # No bound for f
    else:
        bounds = None
        
    # define objective function, c = 0 (fixed)
    def objective(par, y, x, fix):
        par = [fix[i] if fix[i] is not None else par[i] for i in range(len(par))]
        b, d, e, f = par
        y_ = ((d + f*x)/(1 + np.exp(b*(np.log(x + 1e-10)-np.log(e)))))
        return np.sum((y - y_)**2)

    res_ = minimize(objective, par_init, args = (y, x, fix_par), method=opt, bounds=bounds, options = {'maxiter' : 10000})

    # Replace any fixed parameters with their actual values in the result
    res_.x = [float(fix_par[i]) if fix_par[i] is not None else res_.x[i] for i in range(len(res_.x))]
    
    return (res_, "4-parameter Brain-Cousens model", "BC4", ["b", "d", "e", "f"])


# In[ ]:


def Weib1(y, x, par_init = [1, 0, 100, 50], opt = "L-BFGS-B", fix_par = [None, None, None, None]):
    """
    Weibull Type 1 modeling function.

    This function models a response `y` as a function of dose `x` using a Weibull Type 1 model.

    Parameters:
    ----------
    y : array-like
        Response or target variable (dependent variable).
    x : array-like
        Dose or explanatory variable (independent variable).
    par_init : list, optional
        Initial values for the four parameters `b`, `c`, `d`, and `e` to be estimated, in the form [b, c, d, e]. 
        Default is [1, 0, 100, 50].
    opt : str, optional
        Optimization method. Default is "L-BFGS-B", which utilizes bounds. Other methods such as 
        "Nelder-Mead", "TNC", "Powell", and "SLSQP" can also be used, though support for these methods may not be fully verified.
    fix_par : list, optional
        Fixed parameter values for `b`, `c`, `d`, and `e`. If a parameter should be estimated, set it to None.
        Default is [None, None, None, None].

    Returns:
    -------
    tuple
        A tuple with four elements:
        1. Optimization result containing the estimated parameters and other details from the minimization of SSE.
        2. A string indicating the type/name of the model ("Weibull Type 1 Model").
        3. The function name ("Weib1"), useful for future reference.
        4. A list of estimated parameters: ["b", "c", "d", "e"].

    Raises:
    -------
    ValueError:
        If the number of initial parameters in `par_init` does not match the number of fixed parameters in `fix_par`.

    Notes:
    ------
    - The "L-BFGS-B" method is designed for optimization with bounds, and this function provides bounds for 
      the parameter `e` to ensure it is positive.
    """

    # Checking the number of fixed parameters passed
    if len(par_init) != len(fix_par):
        raise ValueError("Mention either 'None' or numerical values to b, c, d, e parameters.")

    # L-BFGS-B can accept bounds. Also, few more optimization methods work well with bounds
    if opt in ["L-BFGS-B", "Nelder-Mead", "TNC", "Powell", "SLSQP"]:
        bounds = [(None, None),  # No bound for b
                  (None, None),  # No bound for c
                  (None, None),  # No bound for d
                  (1e-10, None)] # Bound e to be positive
    else:
        bounds = None
    
    # define objective function: sse
    def objective(par, y, x, fix):
        par = [fix[i] if fix[i] is not None else par[i] for i in range(len(par))]
        b, c, d, e = par
        y_ = c + ((d-c)*np.exp(-np.exp(b*(np.log(x+1e-10)-np.log(e)))))
        return np.sum((y - y_)**2)

    # minimization through sse function
    res_ = minimize(objective, par_init, args = (y, x, fix_par), method=opt, bounds=bounds, options = {'maxiter' : 10000})

    # Replace any fixed parameters with their actual values in the result
    res_.x = [float(fix_par[i]) if fix_par[i] is not None else res_.x[i] for i in range(len(res_.x))]
    
    return (res_, "Weibull Type 1 Model", "Weib1", ["b", "c", "d", "e"])


# In[ ]:


def Weib2(y, x, par_init = [1, 0, 100, 50], opt = "L-BFGS-B", fix_par = [None, None, None, None]):
    """
    Weibull Type 2 modeling function.

    This function models a response `y` as a function of dose `x` using a Weibull Type 2 model.

    Parameters
    ----------
    y : array-like
        Response or target variable (dependent variable).
    x : array-like
        Dose or explanatory variable (independent variable).
    par_init : list, optional
        Initial values for the 4 parameters `b`, `c`, `d`, and `e` to be estimated, in the form [b, c, d, e]. 
        Default is [1, 0, 100, 50].
    opt : str, optional
        Optimization method. Default is "L-BFGS-B", which uses bounds. Other methods such as 
        "Nelder-Mead", "TNC", "Powell", and "SLSQP" can also be used, though support for these methods may not be fully verified.
    fix_par : list, optional
        Fixed parameter values for `b`, `c`, `d`, and `e`. Set a parameter to `None` if it should be estimated.
        Default is [None, None, None, None].

    Returns
    -------
    tuple
        A tuple with four elements:
        1. Optimization result containing the estimated parameters and other details from the minimization of SSE.
        2. A string indicating the type/name of the model ("Weibull Type 2 Model").
        3. The function name ("Weib2"), useful for future reference.
        4. A list of estimated parameters: ["b", "c", "d", "e"].

    Raises
    ------
    ValueError
        If the number of initial parameters in `par_init` does not match the number of fixed parameters in `fix_par`.

    Notes
    -----
    - The "L-BFGS-B" method is designed for optimization with bounds, and this function provides bounds for the parameter `e` to ensure it is positive.
    """

    # Checking the number of fixed parameters passed
    if len(par_init) != len(fix_par):
        raise ValueError("Mention either 'None' or numerical values to b, c, d, e parameters.")

    # L-BFGS-B can accept bounds. Also, few more optimization methods work well with bounds
    if opt in ["L-BFGS-B", "Nelder-Mead", "TNC", "Powell", "SLSQP"]:
        bounds = [(None, None),  # No bound for b
                  (None, None),  # No bound for c
                  (None, None),  # No bound for d
                  (1e-10, None)] # Bound e to be positive
    else:
        bounds = None
    
    # define objective function: sse
    def objective(par, y, x, fix):
        par = [fix[i] if fix[i] is not None else par[i] for i in range(len(par))]
        b, c, d, e = par
        y_ = c + ((d-c)*(1-np.exp(-np.exp(b*(np.log(x+1e-10)-np.log(e))))))
        return np.sum((y - y_)**2)
    
    res_ = minimize(objective, par_init, args = (y, x, fix_par), method=opt, bounds=bounds, options = {'maxiter' : 10000})

    # Replace any fixed parameters with their actual values in the result
    res_.x = [float(fix_par[i]) if fix_par[i] is not None else res_.x[i] for i in range(len(res_.x))]
    
    return (res_, "Weibull Type 2 Model", "Weib2", ["b", "c", "d", "e"])


# In[ ]:


def Gomp4(y, x, par_init = [1, 0, 100, 50], opt = "L-BFGS-B", fix_par = [None, None, None, None]):
    """
    4-Parameter Gompertz modeling function.

    This function models a response `y` as a function of dose `x` using a 4-parameter Gompertz model.

    Parameters:
    ----------
    y : array-like
        Response or target variable (dependent variable).
    x : array-like
        Dose or explanatory variable (independent variable).
    par_init : list, optional
        Initial values of the 4 parameters `b`, `c`, `d`, and `e` to be estimated, in the form [b, c, d, e]. 
        Default is [1, 0, 100, 50].
    opt : str, optional
        Optimization method. Default is "L-BFGS-B", which uses bounds. Other methods such as 
        "Nelder-Mead", "TNC", "Powell", and "SLSQP" can also be used, though support for these methods may not be fully verified.
    fix_par : list, optional
        Fixed parameter values for `b`, `c`, `d`, and `e`. If a parameter should be estimated, set it to None.
        Default is [None, None, None, None].

    Returns:
    -------
    tuple
        A tuple with four elements:
        1. Optimization result containing the estimated parameters and other details from the minimization of SSE.
        2. A string indicating the type/name of the model ("4 Parameter Gompertz Model").
        3. The function name ("Gomp4"), useful for future reference.
        4. A list of estimated parameters: ["b", "c", "d", "e"].

    Raises:
    -------
    ValueError:
        If the number of initial parameters in `par_init` doesn't match the number of fixed parameters in `fix_par`.

    Notes:
    ------
    - The "L-BFGS-B" method is designed for optimization with bounds, and this function provides bounds for the parameter `e` to ensure it's positive.
    """

    # Checking the number of fixed parameters passed
    if len(par_init) != len(fix_par):
        raise ValueError("Mention either 'None' or numerical values to b, c, d, e parameters.")

    # L-BFGS-B can accept bounds. Also, few more optimization methods work well with bounds
    if opt in ["L-BFGS-B", "Nelder-Mead", "TNC", "Powell", "SLSQP"]:
        bounds = [(None, None),  # No bound for b
                  (None, None),  # No bound for c
                  (None, None),  # No bound for d
                  (1e-10, None)] # Bound e to be positive
    else:
        bounds = None
        
    # define objective function
    def objective(par, y, x, fix):
        par = [fix[i] if fix[i] is not None else par[i] for i in range(len(par))]
        b, c, d, e = par
        y_ = c + ((d-c)*(np.exp(-np.exp(b*(x-e)))))
        return np.sum((y - y_)**2)

    # to ignore RuntimeWarning
    warnings.simplefilter("ignore", RuntimeWarning)
    
    res_ = minimize(objective, par_init, args = (y, x, fix_par), method=opt, bounds=bounds, options = {'maxiter' : 10000})

    # Replace any fixed parameters with their actual values in the result
    res_.x = [float(fix_par[i]) if fix_par[i] is not None else res_.x[i] for i in range(len(res_.x))]
    
    return (res_, "4-parameter Gompertz Model", "Gomp4", ["b", "c", "d", "e"])


# In[ ]:


def log_normal(y, x, par_init = [1, 0, 100, 50], opt = "L-BFGS-B", fix_par = [None, None, None, None]):
    """
    Log-normal model fitting function.

    This function models a response `y` as a function of dose `x` using a log-normal function of the form:
    f(x) = c + (d - c) * Φ(b * (log(x) - log(e))), where Φ is the cumulative distribution function (CDF) of the normal distribution.

    Parameters:
    ----------
    y : array-like
        Response or target variable (dependent variable).
    x : array-like
        Dose or explanatory variable (independent variable).
    par_init : list, optional
        Initial values of the 4 parameters `b`, `c`, `d`, and `e` to be estimated, in the form [b, c, d, e]. 
        Default is [1, 0, 100, 50].
    opt : str, optional
        Optimization method. Default is "L-BFGS-B", which uses bounds. Other methods such as 
        "Nelder-Mead", "TNC", "Powell", and "SLSQP" can also be used, though support for these methods may not be fully verified.
    fix_par : list, optional
        Fixed parameter values for `b`, `c`, `d`, and `e`. If a parameter should be estimated, set it to None.
        Default is [None, None, None, None].

    Returns:
    -------
    tuple
        A tuple with four elements:
        1. Optimization result containing the estimated parameters and other details from the minimization of SSE.
        2. A string indicating the type/name of the model ("Log-normal model").
        3. The function name ("log_normal"), useful for future reference.
        4. A list of estimated parameters: ["b", "c", "d", "e"].

    Raises:
    -------
    ValueError:
        If the number of initial parameters in `par_init` doesn't match the number of fixed parameters in `fix_par`.

    Notes:
    ------
    - The "L-BFGS-B" method is designed for optimization with bounds, and this function provides bounds for the parameter `e` to ensure it's positive.
    """

    # Checking the number of fixed parameters passed
    if len(par_init) != len(fix_par):
        raise ValueError("Mention either 'None' or numerical values for b, c, d, e parameters.")

    # L-BFGS-B can accept bounds. Also, few more optimization methods work well with bounds
    if opt in ["L-BFGS-B", "Nelder-Mead", "TNC", "Powell", "SLSQP"]:
        bounds = [(None, None),  # No bound for b
                  (None, None),  # No bound for c
                  (None, None),  # No bound for d
                  (1e-10, None)] # Bound e to be positive
    else:
        bounds = None
    
    # Define the objective function (sum of squared errors, SSE)
    def objective(par, y, x, fix):
        par = [fix[i] if fix[i] is not None else par[i] for i in range(len(par))]
        b, c, d, e = par
        y_ = c + (d - c) * norm.cdf(b * (np.log(x + 1e-10) - np.log(e)))
        return np.sum((y - y_)**2)
    
    # Perform the minimization
    res_ = minimize(objective, par_init, args=(y, x, fix_par), method=opt, bounds=bounds, options={'maxiter': 10000})
    
    # Replace any fixed parameters with their actual values in the result
    res_.x = [float(fix_par[i]) if fix_par[i] is not None else res_.x[i] for i in range(len(res_.x))]
    
    return (res_, "Log-normal model", "log_normal", ["b", "c", "d", "e"])


# In[ ]:




