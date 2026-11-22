.PHONY: setup run serve similar quality check-data load try-frame-rate try-index-types try-quality-thresholds smoke embeddings analyze clean

# A bare `make` runs the smoke test, as it did before the uniform targets were added.
.DEFAULT_GOAL := smoke

# Uniform targets for The Playbooks. The book prints no commands for this bonus playbook. The
# README's commands (./install.sh or pip install -r requirements.txt,
# ./run_pipeline.sh --generate-embeddings, --start-api, --start-dashboard, --quality-check,
# python analyze_embeddings.py) still work as written.

API_PORT ?= 8085
DASHBOARD_PORT ?= 8095

# Install the locked dependencies into .venv. The ResNet-18 weights (about 45 MB) download on the
# first embedding run.
setup:
	uv sync --frozen

# Generate embeddings for the 32 bundled videos and 7 reference images, then print the analysis
# and the similarity search results. CPU only; exits when done.
run: setup
	@$(MAKE) --no-print-directory embeddings analyze

# Start the similarity search API (http://localhost:8085) and the dashboard
# (http://localhost:8095). Runs until Ctrl-C. Generates the embeddings first if they are missing.
serve: setup
	@test -d data/embeddings || $(MAKE) --no-print-directory embeddings
	uv run --frozen bash -c 'PORT=$(API_PORT) python src/api/main.py & api=$$!; \
		trap "kill $$api 2>/dev/null" EXIT; \
		DASHBOARD_PORT=$(DASHBOARD_PORT) API_URL=http://localhost:$(API_PORT) python src/api/dashboard.py'

# With the API running (make serve): the closest videos and photos to one, and a quality check.
#   make similar ID="download (25)"
#   make quality VIDEO="download (25)" IMAGE=Salmon_Edamame.jpg THRESHOLD=0.8
similar:
	@test -n "$(ID)" || { echo 'Usage: make similar ID="download (25)"'; exit 2; }
	uv run --frozen python scripts/ask_api.py similar "$(ID)"

quality:
	@test -n "$(VIDEO)" -a -n "$(IMAGE)" || { echo 'Usage: make quality VIDEO="download (25)" IMAGE=Salmon_Edamame.jpg [THRESHOLD=0.8]'; exit 2; }
	uv run --frozen python scripts/ask_api.py quality "$(VIDEO)" "$(IMAGE)" --threshold $(or $(THRESHOLD),0.75)

# Your own videos and photos: videos/ and images/ in DIR. check-data checks every file and writes
# nothing; load adds their embeddings to data/embeddings and lists the closest matches.
#   make check-data DIR=my_media
#   make load DIR=my_media
check-data:
	@test -n "$(DIR)" || { echo "Usage: make check-data DIR=path/to/folder"; exit 2; }
	uv run --frozen python scripts/own_media.py --dir "$(DIR)" --check-only

load:
	@test -n "$(DIR)" || { echo "Usage: make load DIR=path/to/folder"; exit 2; }
	uv run --frozen python scripts/own_media.py --dir "$(DIR)"

# Try this: three measured experiments (see README, "Try this"). CPU only, no Docker.
# The first two need the embeddings from make run.
try-frame-rate: setup
	uv run --frozen python experiments/frame_rate.py

try-index-types: setup
	uv run --frozen python experiments/index_types.py

try-quality-thresholds: setup
	uv run --frozen python experiments/quality_thresholds.py

# Sample data, embeddings on a subset, FAISS search, the APIs, the quality tool, the dashboard. CPU only.
smoke:
	uv run --frozen pytest -q tests/test_smoke.py

# All 32 videos and 7 images (about 4-5 minutes on a laptop CPU), then the analysis
embeddings:
	uv run --frozen python src/embeddings/generator.py --input data/videos,data/images --output data/embeddings

analyze:
	uv run --frozen python analyze_embeddings.py
	uv run --frozen python test_similarity_search.py

# Remove generated embeddings and quality reports
clean:
	rm -rf data/embeddings data/quality
