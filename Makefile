.PHONY: install

install:
	@pipenv install

test:
	python3 tests/test_lossfunction.py

run:
	python3 ./src/process_data.py

clean:
	- rm -rf *.db
	- rm -rf *.html
	- rm output/*.db
	- rm output/*.html