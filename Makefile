.PHONY:

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = filtered-vector-search-demo

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## start qdrant
qdrant_start:
	docker compose up -d --build

## stop qdrant
qdrant_stop:
	docker compose stop

## lint and formatting
lint:
	poetry run isort src
	poetry run black src -l 79
	poetry run flake8 src 

## run api backend
api:
	cd src/api ; poetry run uvicorn main:app --reload

## run ui
ui:
	poetry run python -m src/visualization/visualize.py

## run pipeline
repro: check_commit PIPELINE.md
	poetry run dvc repro
	git commit dvc.lock -m '[update] dvc repro' || true

## check commit
check_commit:
	git diff --exit-code --staged
	git diff --exit-code

## PIPELINE.md
PIPELINE.md: dvc.yaml params.yaml
	poetry run dvc dag --md > PIPELINE.md
	git commit PIPELINE.md -m '[update] dvc pipeline update' || true

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
