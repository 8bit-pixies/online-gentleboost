render:
    @quarto render paper/paper.qmd

lint:
	@isort . --check-only 
	@ruff check . --fix
	@black --check .
	@echo "lint finished"

format:
	@isort .
	@black .