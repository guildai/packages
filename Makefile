pkg-all:
	for pkg in `find gpkg -name Makefile | xargs dirname`; do \
	  if [ -e $$pkg/Makefile ]; then make -C $$pkg pkg; fi \
	done

upload-all:
	for pkg in `find gpkg -name Makefile | xargs dirname`; do \
	  if [ -e $$pkg/Makefile ]; then make -C $$pkg upload; fi \
	done

test-all:
	for pkg in `find gpkg -name Makefile | xargs dirname`; do \
	  if [ -e $$pkg/Makefile ]; then \
            make -C $$pkg test || test "$$?" = "0" -o "$$?" = "2" ; \
          fi \
	done

clean-all:
	for pkg in `find gpkg -name Makefile | xargs dirname`; do \
	  if [ -e $$pkg/Makefile ]; then make -C $$pkg clean; fi \
	done

clean-test-all:
	for pkg in `find gpkg -name Makefile | xargs dirname`; do \
	  if [ -e $$pkg/Makefile ]; then make -C $$pkg clean-test ; fi \
	done
