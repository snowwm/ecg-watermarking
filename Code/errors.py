class WMError(Exception):
    msg = None

    def __init__(self, msg=None, suffix=None) -> None:
        if msg is None:
            msg = self.msg
        if suffix is not None:
            msg += f" ({suffix})"
        super().__init__(msg)


class InvalidConfig(WMError):
    msg = "Invalid embedder/extractor configuration"


class InsufficientContainerRange(WMError):
    msg = "Insufficient container range"


class CantEmbed(WMError):
    msg = "Insufficient container length or range for this watermark"


class CantExtract(WMError):
    msg = "Could not find watermark with given length"
