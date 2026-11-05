Real bugs and surprises encountered while building AI Run Tracker & Monitor:

- `random.Random.random()` takes no arguments; generate defect masks with a list comprehension instead.
- No `pip` installed and PEP 668 externally-managed environment; bootstrapped pip via get-pip.py with --break-system-packages.
- pandas `DataFrame.sample(random_state=)` rejects a `random.Random` object; seed with an int instead.
- `datetime.utcnow()` is deprecated in Python 3.12; switched to `datetime.now(timezone.utc)`.
- `dq_warning_count` referenced before assignment in train.py caused UnboundLocalError; fixed declaration order.
- MLflow logs a warning when a model is logged without a signature or input example.
- Docker Compose `docker compose wait` is not a real command; removed it from demo.sh.
- Shared volume: dashboard mounts read-only while the pipeline writes runs.db/reports to the same volume.
