from hgf.utils import *
from hgf.exceptions import ParameterConfigurationError, ModelConfigurationError

class Parameter(object):
    """Parameters of nodes"""
    def __init__(self,
                 *,
                 space='native',
                 lower_bound=None,
                 upper_bound=None,
                 value=None,
                 trans_value=None,
                 prior_mean=None,
                 trans_prior_mean=None,
                 trans_prior_precision=None):

        # Initialize attributes
        self.space = space

        if lower_bound is not None:
            self.lower_bound = lower_bound

        if upper_bound is not None:
            self.upper_bound = upper_bound

        if value is not None and trans_value is not None:
            raise ParameterConfigurationError(
                'Only one of value and trans_value can be given.')
        elif value is not None:
            self.value = value
        elif trans_value is not None:
            self.trans_value = trans_value
        else:
            raise ParameterConfigurationError(
                'One of value and trans_value must be given.')

        if prior_mean is not None and trans_prior_mean is not None:
            raise ParameterConfigurationError(
                'Only one of prior_mean and trans_prior_mean can be given.')
        elif prior_mean is not None:
            self.prior_mean = prior_mean
        else:
            self.trans_prior_mean = trans_prior_mean

        if (trans_prior_precision is None and
            (prior_mean is not None or
             trans_prior_mean is not None)):
            raise ParameterConfigurationError(
                'trans_prior_precision must be given if prior_mean ' +
                'or trans_prior_mean is given')
        else:
            self.trans_prior_precision = trans_prior_precision

    @property
    def space(self):
        return self._space

    @space.setter
    def space(self, space):
        if space is 'native':
            self._space = space
            self._lower_bound = None
            self._upper_bound = None
        elif space is 'log':
            self._space = space
            self._lower_bound = 0
            self._upper_bound = None
        elif space is 'logit':
            self._space = space
            self._lower_bound = 0
            self._upper_bound = 1
        else:
            raise ParameterConfigurationError(
                "Space must be one of 'native, 'log', or 'logit'")

        # Recalculate trans_value
        try:
            self.value = self._value
        except AttributeError:
            pass
        # Recalculate trans_prior_mean
        try:
            self.prior_mean = self._prior_mean
        except AttributeError:
            pass

    @property
    def lower_bound(self):
        return self._lower_bound

    @lower_bound.setter
    def lower_bound(self, lower_bound):
        space = self.space
        if lower_bound is not None and space is 'native':
            raise ParameterConfigurationError(
                "lower_bound must be None if space is 'native'.")
        elif lower_bound is not None and lower_bound > self.value:
            raise ParameterConfigurationError(
                'lower_bound may not be greater than current value')
        elif space is 'log':
            self._lower_bound = lower_bound
            self._upper_bound = None
        else:
            self._lower_bound = lower_bound

        # Recalculate trans_value
        try:
            self.value = self._value
        except AttributeError:
            pass
        # Recalculate trans_prior_mean
        try:
            self.prior_mean = self._prior_mean
        except AttributeError:
            pass

    @property
    def upper_bound(self):
        return self._upper_bound

    @upper_bound.setter
    def upper_bound(self, upper_bound):
        space = self.space
        if upper_bound is not None and space is 'native':
            raise ParameterConfigurationError(
                "upper_bound must be None if space is 'native'.")
        elif upper_bound is not None and upper_bound < self.value:
            raise ParameterConfigurationError(
                'upper_bound may not be less than current value')
        elif space is 'log':
            self._lower_bound = None
            self._upper_bound = upper_bound
        else:
            self._upper_bound = upper_bound

        # Recalculate trans_value
        try:
            self.value = self._value
        except AttributeError:
            pass
        # Recalculate trans_prior_mean
        try:
            self.prior_mean = self._prior_mean
        except AttributeError:
            pass

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if self.lower_bound is not None and value < self.lower_bound:
            raise ParameterConfigurationError(
                'value may not be less than current lower_bound')
        elif self.upper_bound is not None and value > self.upper_bound:
            raise ParameterConfigurationError(
                'value may not be greater than current upper_bound')
        else:
            self._value = value

        space = self.space
        if space is 'native':
            self._trans_value = value
        elif space is 'log':
            self._trans_value = log(value,
                                    lower_bound=self.lower_bound,
                                    upper_bound=self.upper_bound)
        elif space is 'logit':
            self._trans_value = logit(value,
                                      lower_bound=self.lower_bound,
                                      upper_bound=self.upper_bound)

    @property
    def trans_value(self):
        return self._trans_value

    @trans_value.setter
    def trans_value(self, trans_value):
        self._trans_value = trans_value

        space = self.space
        if space is 'native':
            self._value = trans_value
        elif space is 'log':
            self._value = exp(trans_value,
                              lower_bound=self.lower_bound,
                              upper_bound=self.upper_bound)
        elif space is 'logit':
            self._value = sgm(trans_value,
                              lower_bound=self.lower_bound,
                              upper_bound=self.upper_bound)

    @property
    def prior_mean(self):
        return self._prior_mean

    @prior_mean.setter
    def prior_mean(self, prior_mean):
        self._prior_mean = prior_mean

        if prior_mean is not None:
            space = self.space
            if space is 'native':
                self._trans_prior_mean = prior_mean
            elif space is 'log':
                self._trans_prior_mean = log(prior_mean,
                                             lower_bound=self.lower_bound,
                                             upper_bound=self.upper_bound)
            elif space is 'logit':
                self._trans_prior_mean = logit(prior_mean,
                                               lower_bound=self.lower_bound,
                                               upper_bound=self.upper_bound)
        else:
            self._trans_prior_mean = None
            self._trans_prior_precision = None

    @property
    def trans_prior_mean(self):
        return self._trans_prior_mean

    @trans_prior_mean.setter
    def trans_prior_mean(self, trans_prior_mean):
        self._trans_prior_mean = trans_prior_mean

        if trans_prior_mean is not None:
            space = self.space
            if space is 'native':
                self._prior_mean = trans_prior_mean
            elif space is 'log':
                self._prior_mean = exp(trans_prior_mean,
                                       lower_bound=self.lower_bound,
                                       upper_bound=self.upper_bound)
            elif space is 'logit':
                self._prior_mean = sgm(trans_prior_mean,
                                       lower_bound=self.lower_bound,
                                       upper_bound=self.upper_bound)
        else:
            self._prior_mean = None
            self._trans_prior_precision = None

    @property
    def trans_prior_precision(self):
        return self._trans_prior_precision

    @trans_prior_precision.setter
    def trans_prior_precision(self, trans_prior_precision):
        self._trans_prior_precision = trans_prior_precision

        if trans_prior_precision is None:
            self._prior_mean = None
            self._trans_prior_mean = None

    def log_prior(self):
        try:
            return -gaussian_surprise(self.trans_value,
                                      self.trans_prior_mean,
                                      self.trans_prior_precision)
        except AttributeError as e:
            raise ModelConfigurationError(
                'trans_prior_mean and trans_prior_precision attributes ' +
                'must\nbe specified for method log_prior to return ' +
                'a value.') from e

