# drm_basic Library

In Python, the availability and support for Dose-Response (DR) related methods have been given less importance compared to R, where statisticians have implemented numerous DR methods that are well-maintained. The primary motivation behind this library is to address the lack of DR implementations in Python, particularly in academia.

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
To use the `drm_basic` library, first, you need to import it into your Python script. Here's how you can do that:

```python
import drm_basic
```


