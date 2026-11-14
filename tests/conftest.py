import os
os.environ["LLM_MODE"] = "mock"
os.environ["RATE_LIMIT"] = "1000/minute"
os.environ.setdefault("DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))
