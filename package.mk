test_env_dir := test-env

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
	$(root)/test-package "$(root)" "$(test_env_dir)"

clean-test:
	rm -rf $(test_env_dir)
