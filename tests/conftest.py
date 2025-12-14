import warnings

warnings.filterwarnings(
    "ignore",
    message=r'Field "model_(path|spec)" has conflict with protected namespace "model_".',
    category=UserWarning,
)

warnings.filterwarnings(
    "ignore",
    message=r"datetime\.datetime.utcnow\(\) is deprecated",
    category=DeprecationWarning,
)
