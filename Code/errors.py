class WMError(Exception):
    msg = None

    def __init__(self, msg=None, suffix=None) -> None:
        if msg is None:
            msg = self.msg
        if suffix is not None:
            msg += f" ({suffix})"
        super().__init__(msg)


class StaticError(WMError):
    """These errors are raised before actual embedding/extraction starts."""


class DynamicError(WMError):
    """These errors are raised in the course of embedding/extraction."""


class InvalidConfig(StaticError):
    msg = "Invalid algorithm configuration"


class InsufficientContainerRangeStatic(StaticError):
    msg = "Insufficient container range"


class CantEmbed(DynamicError):
    msg = "Insufficient container length or range for this watermark"


class InsufficientContainerRangeDynamic(CantEmbed):
    msg = "Insufficient container range"


class CantExtract(DynamicError):
    msg = "Could not find watermark with given length"
