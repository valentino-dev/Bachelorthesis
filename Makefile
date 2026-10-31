.PHONY: run test lat


run:
	cd src;\
		python main.py


test:
	cd src/tests;\
		python main.py

lat:
	cd src;\
		python lattice.py
