# So `make` alone starts the app. Nothing else is required.
.PHONY: run setup test
run:
	@./run

# Build the venv and install dependencies without starting the server.
setup:
	@./run --setup-only

# Depends on setup because test_app.py imports numpy/skimage, which live in the venv --
# on a fresh clone `python3` has none of them. Waits for the port to answer instead of
# sleeping a fixed number of seconds: the first run installs dependencies for minutes,
# and the old `sleep 25` raced it and reported the whole suite as connection failures.
test: setup
	@OPEN=0 PORT=8799 ./run & SRV=$$!; \
	 for i in $$(seq 1 180); do \
	   curl -sf -o /dev/null http://127.0.0.1:8799/ && break; \
	   kill -0 $$SRV 2>/dev/null || { echo "server exited before it served"; exit 1; }; \
	   sleep 1; \
	 done; \
	 BASE=http://127.0.0.1:8799 ./.venv/bin/python3 \
	   interior_active_learning/code/test_app.py; RC=$$?; \
	 kill $$SRV 2>/dev/null || true; wait $$SRV 2>/dev/null || true; exit $$RC
