# drm_basic Library

In Python, the availability and support for Dose-Response (DR) related methods have been given less importance compared to R, where statisticians have implemented numerous DR methods that are well-maintained. The primary motivation behind this library is to address the lack of DR implementations in Python, particularly in academia.

The methods available in this library currently work with continuous response data. Future updates will include support for Poisson and binomial response data. The default optimization algorithm used is `L-BFGS-B`, but other optimization methods can also be applied. For details on how to use alternative optimizers, refer to each function's documentation (accessible with Shift+Tab in Windows).

This library implements basic dose-response modeling methods. There are 10 different dose-response functions available, as listed below:

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

## Installation

You can install the package directly from the GitHub repository using the following command:

```bash
pip install git+https://github.com/gautampk95/drm_basic.git
```

Alternatively, you can install it through the unzipped package file:

```bash
pip install drm_basic
```

## Usage
To use the `drm_basic` library, you first need to import it into your Python script. Here's how you can do that:

```python
import drm_basic
```
To import the dose-response modeling methods, use the following:
```python
from drm_basic import drm_methods
```
To import available utility functions, use the following:
```python
from drm_basic import drm_utils
```

## Example 1: Using Methods from drm_methods and drm_utils
Here’s a simple example of how to use a method from the drm_methods module:
```python
# Example data (arrays or dataframe columns)
import numpy as np

dose = np.array([0, 1, 2, 3, 4, 5])
response = np.array([0, 10, 30, 50, 70, 90])

# Fit a model using a method from drm_methods
model = drm_methods.LL4(response, dose)

# To check the summary of the model
drm_utils.drm_summary(model)

# If a test data:(dose_test, response_test) is available, the model is fitted as shown
dose_test = np.array([1.1, 2.1, 5.1])
response_test = np.array([11, 32, 92])
response_pred = drm_utils.drm_predict(model, dose_test)

# To get/view the fitted parameters
params = model[0].x
print("Fitted parameters for LL4 model:", params)
```
The performance of the model can be found:
```python
# RMSE is calculated
from sklearn.metrics import mean_squared_error, mean_absolute_error
rmse = np.sqrt(mean_squared_error(response_test, response_pred))
print("The RMSE value:", rmse)
```




