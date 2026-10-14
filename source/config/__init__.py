"""Configuration (``settings.py``): paths, model, tracker, worker and API defaults."""
from source.config.settings import (APP, DETECTOR, MODELS_DIR, OUTPUT_DIR, PROJECT_ROOT, REID,
                                    SOURCE_DIR, TRACKER, UPLOAD_DIR, AppConfig, DetectorConfig,
                                    ReIDConfig, TrackerConfig, as_dict, baseline_tracker_config)

__all__ = ["APP", "DETECTOR", "MODELS_DIR", "OUTPUT_DIR", "PROJECT_ROOT", "REID", "SOURCE_DIR",
           "TRACKER", "UPLOAD_DIR", "AppConfig", "DetectorConfig", "ReIDConfig", "TrackerConfig",
           "as_dict", "baseline_tracker_config"]
