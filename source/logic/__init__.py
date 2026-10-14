"""Shared business logic used by both the API (``source/main.py``) and ``worker.py``.

* ``video_processor``   decode -> detect -> embed -> associate -> render -> write
* ``inference_manager`` automatic, verified ReID backend selection + runtime fallback
* ``job_manager``       job queue, lifecycle, persistence and cleanup
* ``tracking``          Kalman filter, association, track lifecycle, ReID crop extraction
"""
