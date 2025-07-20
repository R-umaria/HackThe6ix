from enum import Enum

class DrowsinessTier(Enum):
    NONE = "None"
    LOW = "low"
    MEDIUM_LOW = "medium low"
    MEDIUM = "medium"
    MEDIUM_HIGH = "medium high"
    HIGH = "high"

drowsiness_tier = DrowsinessTier.NONE