# So `make` alone starts the app. Nothing else is required.
.PHONY: run test
run:
	@./run
test:
	@PORT=8799 ./run & sleep 25; BASE=http://127.0.0.1:8799 \
	  python3 interior_active_learning/code/test_app.py; kill %1 2>/dev/null || true
