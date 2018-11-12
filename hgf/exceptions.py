class HgfException(Exception):
    """Base class for all exceptions raised by the hgf module."""


class ModelConfigurationError(HgfException):
    """Model configuration error."""


class NodeConfigurationError(HgfException):
    """Node configuration error."""


class ParameterConfigurationError(HgfException):
    """Parameter configuration error."""


class HgfUpdateError(HgfException):
    """Error owing to a violation of the assumptions underlying HGF updates."""


class OutcomeValueError(HgfException):
    """Outcome value error."""


class LogArgumentError(HgfException):
    """Log argument out of bounds."""


class LogitArgumentError(HgfException):
    """Logit argument out of bounds."""
