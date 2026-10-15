.PHONY: install test demo data eval eval-cheap bench api agent docker lint

install:            ## batch + api + eval deps
	pip install -r requirements.txt && pip install -e . --no-deps

test:               ## unit tests (no keys, no network)
	python -m unittest discover -s tests -v

demo:               ## full pipeline + metrics on the synthetic fixture, no keys
	python scripts/demo_offline.py

data:               ## AMI annotations + dev-mini audio
	python scripts/download_ami.py --split dev-mini

eval:               ## full eval incl. LLM judge on dev-mini
	python scripts/run_eval.py --split dev-mini

eval-cheap:         ## WER / cpWER / DER only on the test split
	python scripts/run_eval.py --split test --no-summary

bench:              ## real-time latency benchmark on one meeting
	python scripts/bench_realtime.py data/audio/ES2011a.Mix-Headset.wav --seconds 300 --turns 10

api:
	uvicorn convintel.api.main:app --reload --port 8000

agent:
	python -m convintel.realtime.agent dev

docker:
	docker compose build

lint:
	ruff check src scripts tests
