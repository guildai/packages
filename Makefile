pkg-all:
	for pkg in `find -name Makefile | xargs dirname`; do \
	  if [ -e $$pkg/Makefile ]; then make -C $$pkg pkg; fi \
	done

upload-all:
	for pkg in `find -name Makefile | xargs dirname`; do \
	  if [ -e $$pkg/Makefile ]; then make -C $$pkg upload; fi \
	done

test-all:
	for pkg in `find -name Makefile | xargs dirname`; do \
	  if [ -e $$pkg/Makefile ]; then make -C $$pkg test; fi \
	done

clean-all:
	for pkg in `find -name Makefile | xargs dirname`; do \
	  if [ -e $$pkg/Makefile ]; then make -C $$pkg clean; fi \
	done
