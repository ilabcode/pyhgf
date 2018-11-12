def exp(x, *, lower_bound=0, upper_bound=None):
    """The (shifted and mirrored) exponential function"""
    if upper_bound is not None:
        return -np.exp(x) + upper_bound
    else:
        return np.exp(x) + lower_bound


def log(x, *, lower_bound=0, upper_bound=None):
    """The (shifted and mirrored) natural logarithm"""
    if upper_bound is not None:
        if x > upper_bound:
            raise LogArgumentError('Log argument may not be greater than ' +
                                   'upper bound.')
        elif x == upper_bound:
            return -np.inf
        else:
            return np.log(-x + upper_bound)
    else:
        if x < lower_bound:
            raise LogArgumentError('Log argument may not be less than ' +
                                   'lower bound.')
        elif x == lower_bound:
            return -np.inf
        else:
            return np.log(x - lower_bound)


def sgm(x, *, lower_bound=0, upper_bound=1):
    """The logistic sigmoid function"""
    return (upper_bound - lower_bound) / (1 + np.exp(-x)) + lower_bound


def logit(x, *, lower_bound=0, upper_bound=1):
    """The logistic function"""
    if x < lower_bound:
        raise LogitArgumentError('Logit argmument may not be less than ' +
                                 'lower bound.')
    elif x > upper_bound:
        raise LogitArgumentError('Logit argmument may not be greater than ' +
                                 'upper bound.')
    elif x == lower_bound:
        return -np.inf
    elif x == upper_bound:
        return np.inf
    else:
        return np.log((x - lower_bound) / (upper_bound - x))


def gaussian(x, mu, pi):
    """The Gaussian density as defined by mean and precision"""
    return pi / np.sqrt(2 * np.pi) * np.exp(-pi / 2 * (x - mu)**2)


def gaussian_surprise(x, muhat, pihat):
    """Surprise at an outcome under a Gaussian prediction"""
    return 0.5 * (np.log(2 * np.pi) - np.log(pihat) + pihat * (x - muhat)**2)


def binary_surprise(x, muhat):
    """Surprise at a binary outcome"""
    if x == 1:
        return -np.log(1 - muhat)
    if x == 0:
        return -np.log(muhat)
    else:
        raise OutcomeValueError('Outcome needs to be either 0 or 1.')


