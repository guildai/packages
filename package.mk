pkg: clean
	guild package

clean:
	rm -rf dist build *.egg-info

upload: clean
	export TWINE_USERNAME=guildai; \
	export TWINE_PASSWORD=`gpg --quiet --batch -d $(root)/.pypi-creds.gpg`; \
	guild package --upload --skip-existing

.PHONY: test

test:
	@test ! -e test/run && \
	  echo "No tests found (create test/run to run tests for " \
               "this package)" || test/run
