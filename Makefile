default:
	@echo "This Makfile doesn't have a default target"

package-all: clean
	set -eu; \
	for f in `find -name .package`; do \
	  echo "==== $$f ===="; \
	  (cd $$(dirname $$f) && guild package); \
	done

package-and-upload-all: clean
	set -eu; \
	export TWINE_USERNAME=guildai; \
	export TWINE_PASSWORD=`gpg --quiet --batch -d .pypi-creds.gpg`; \
	for f in `find -name .package`; do \
	  echo "==== $$f ===="; \
	  (cd $$(dirname $$f) && guild package --upload --skip-existing); \
	done

clean:
	find -name dist -type d | xargs -r rm -rf
	find -name build -type d | xargs -r rm -rf
	find -name *egg-info -type d | xargs -r rm -rf
