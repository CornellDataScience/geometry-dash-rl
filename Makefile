.PHONY: setup teacher student

setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -U pip && pip install -r requirements.txt && pip install -e .
	mkdir -p artifacts

teacher:
	. .venv/bin/activate && python -m gdrl.teacher.train_ppo

student:
	. .venv/bin/activate && python -m gdrl.student.train_distill_dataset
