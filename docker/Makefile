
container:
	docker build -t deep_learning_exploration_base -f Dockerfile ..

test_nvidia:
	nvidia-docker run  nvidia/cuda:10.0-base nvidia-smi
