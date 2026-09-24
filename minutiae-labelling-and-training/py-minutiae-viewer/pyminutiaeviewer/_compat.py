# overrides>=4 enforces signature/return-type compatibility at import time, and
# this 2019 code's type hints (e.g. `-> Image` meaning the PIL module) fail it.
# The decorator is documentation only here, so a no-op keeps behaviour identical.
def overrides(method):
    return method
