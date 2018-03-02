pkg-all:
	for pkg in `find -name .package | xargs dirname`; do \
	  if [ -e $$pkg/Makefile ]; then make -C $$pkg pkg; fi \
	done

upload-all:
	for pkg in `find -name .package | xargs dirname`; do \
	  if [ -e $$pkg/Makefile ]; then make -C $$pkg upload; fi \
	done

clean-all:
	for pkg in `find -name .package | xargs dirname`; do \
	  if [ -e $$pkg/Makefile ]; then make -C $$pkg clean; fi \
	done
