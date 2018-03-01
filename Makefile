package-and-upload-all:
	set -eu; \
	export TWINE_USERNAME=guildai; \
	export TWINE_PASSWORD=`gpg --quiet --batch -d .pypi-creds.gpg`; \
	for f in `find -name PACKAGE | grep -v "egg-info/\|/build/"`; do \
	  (cd $$(dirname $$f) && guild package --upload --skip-existing); \
	done
