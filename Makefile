.PHONY: setup mod train train-sm eval smoke clean

setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -U pip && pip install -r requirements.txt && pip install tensorboard && pip install -e .
	mkdir -p artifacts

mod:
	cmake --build mods/TrainingPipeline/build --config Release

train:
	. .venv/bin/activate && python -m gdrl.train.dqn_cv --levels stereo_madness back_on_track polargeist

train-sm:
	. .venv/bin/activate && python -m gdrl.train.dqn_cv --levels stereo_madness --no-curriculum

eval:
	. .venv/bin/activate && python -m gdrl.eval.live_eval --checkpoint artifacts/dqn_cv/latest.pt --episodes 5

smoke:
	. .venv/bin/activate && python -m gdrl.scripts.smoke_frame

clean:
	rm -rf artifacts/dqn_cv/__pycache__
	find src -name '__pycache__' -prune -exec rm -rf {} +
