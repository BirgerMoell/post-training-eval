.PHONY: check site test

check:
	python3 -m post_training_eval.cli validate

site:
	python3 -m post_training_eval.cli site

test:
	python3 -m unittest discover -s tests -v

