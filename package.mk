test_env_dir := test-env
test_op := _check:all

pkg: clean
	guild package

clean:
	rm -rf dist build *.egg-info

upload: clean
	bash -c 'source <( gpg -d $(root)/.pypi-creds.gpg ); guild package --upload --skip-existing'

.PHONY: test

test:
	$(root)/test-package "$(root)" "$(test_env_dir)" "$(test_op)"

clean-test:
	rm -rf $(test_env_dir)

lint:
	PYTHONPATH=.:$(root)/../guild pylint -rn -f parseable *.py
