test_env_dir := test-env
test_op := _check:all

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
	$(root)/test-package "$(root)" "$(test_env_dir)" "$(test_op)"

clean-test:
	rm -rf $(test_env_dir)

lint:
	PYTHONPATH=.:$(root)/../guild pylint -rn -f parseable *.py
