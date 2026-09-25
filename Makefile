# So `make` alone starts the app. Nothing else is required.
.PHONY: run setup test test-browser test-all
run:
	@./run

# Build the venv and install dependencies without starting the server.
setup:
	@./run --setup-only

# Depends on setup because test_app.py imports numpy/skimage, which live in the venv --
# on a fresh clone `python3` has none of them. Waits for the port to answer instead of
# sleeping a fixed number of seconds: the first run installs dependencies for minutes,
# and the old `sleep 25` raced it and reported the whole suite as connection failures.
# Refuse to start if 8799 is already answering. The wait loop below polls the port, and a
# port cannot say which checkout is behind it: a server left over from another run -- another
# clone, a second `make test`, a stray background job -- satisfies the loop immediately, and
# the whole suite then measures THAT tree while reporting on this one. Observed 2026-09-24,
# when a clean-clone run and a local run were started together: they fought over the port and
# the clone's run died 15 checks in, which was the lucky outcome. The unlucky one is a green
# run against the wrong code.
test: setup
	@if curl -sf -o /dev/null -m 2 http://127.0.0.1:8799/ 2>/dev/null; then \
	   echo "port 8799 is already serving something. This target would measure it instead"; \
	   echo "of this checkout, and report a green run against the wrong code."; \
	   echo "Stop it first:  lsof -ti :8799 | xargs kill"; \
	   exit 1; \
	 fi
	@OPEN=0 PORT=8799 ./run & SRV=$$!; \
	 for i in $$(seq 1 180); do \
	   curl -sf -o /dev/null http://127.0.0.1:8799/ && break; \
	   kill -0 $$SRV 2>/dev/null || { echo "server exited before it served"; exit 1; }; \
	   sleep 1; \
	 done; \
	 BASE=http://127.0.0.1:8799 ./.venv/bin/python3 \
	   interior_active_learning/code/test_app.py; RC=$$?; \
	 kill $$SRV 2>/dev/null || true; wait $$SRV 2>/dev/null || true; exit $$RC

# The claim registry. Recomputes every statistic quoted in the analysis docs from its source
# artefact, and checks that named sentences still appear verbatim in the documents that carry
# them. It was written to be run and then never wired to anything -- it appeared in no Makefile
# target, no CI job and no hook, so "the claims are checked" meant "somebody remembered to run
# it". Exits 0 on a fresh clone: artefacts that are gitignored and regenerable report SKIP, and
# an absent artefact is not a drifted number.
verify-claims: setup
	@cd crack_export && ../.venv/bin/python3 tools/verify_claims.py

# The browser test. Separate from `test` on purpose: it needs playwright plus a ~150 MB
# chromium download, which is too much to impose on someone who just wants to run the app.
# It starts its own server against empty scratch directories and touches no repo data.
#   ./.venv/bin/python3 -m pip install -r requirements-dev.txt
#   ./.venv/bin/python3 -m playwright install chromium
test-browser:
	@./.venv/bin/python3 interior_active_learning/code/test_browser.py

# Everything. test_app.py covers the server, test_browser.py covers what only a browser can
# see -- a stroke going to the wrong image's mask was invisible to 326 server-side checks.
test-all: test test-browser
