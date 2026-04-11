.PHONY: setup teacher teacher-mock student smoke test

setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -U pip && pip install -r requirements.txt && pip install -e .
	mkdir -p artifacts

teacher:
	. .venv/bin/activate && python -m gdrl.teacher.train_ppo

teacher-mock:
	. .venv/bin/activate && python -m gdrl.teacher.train_ppo \
		--env-mode mock \
		--total-timesteps 200000 \
		--n-envs 4 \
		--norm-obs --norm-reward \
		--eval-freq 10000 \
		--out artifacts/teacher_ppo_mock \
		--checkpoint-dir artifacts/checkpoints_mock

student:
	. .venv/bin/activate && python -m gdrl.student.train_distill_dataset

smoke:
	. .venv/bin/activate && PYTHONPATH=src python -m gdrl.env.test_mock_env

test:
	. .venv/bin/activate && PYTHONPATH=src python -m gdrl.env.test_mock_env
	. .venv/bin/activate && PYTHONPATH=src python -m unittest discover -s tests -v
