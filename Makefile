VENV_PATH := $(abspath venv)
PIP_REQUIREMENTS:=$(abspath requirements.in)
PCP_LOG_LIBRARY:=$(abspath ./dist/pcplog-1.0.0-py3-none-any.whl)
GNUPLOT_INSTALLED:=$(shell gnuplot --version 2> /dev/null)
PYTHON_INSTALLED := $(shell python3.6 --version 2> /dev/null)
VENV_EXISTS:=$(shell ls ${VENV_PATH} 2> /dev/null)

install-python:
ifndef PYTHON_INSTALLED
	apt-get -y install python3.6
	python3.6 -m pip install pip
endif
.PHONY: install-python

create-venv: install-python
ifndef VENV_EXISTS
	python3.6 -m virtualenv ${VENV_PATH};)
endif
.PHONY: create-venv


prepare-venv: create-venv
	${VENV_PATH}/bin/pip3.6 install ${PCP_LOG_LIBRARY}
	${VENV_PATH}/bin/pip3.6 install -r ${PIP_REQUIREMENTS}
.PHONY: prepare-venv

clean:
	rm -Rf ${VENV_PATH}
.PHONY: clean


