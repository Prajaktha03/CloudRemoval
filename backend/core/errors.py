class InputValidationError(ValueError):
    """Raised when uploaded imagery is malformed or incompatible with the model."""


class ModelUnavailableError(RuntimeError):
    """Raised when inference is requested without a usable trained model."""
