#!/bin/bash
PYTHONPATH=../../proto_compiled:../../ python3 $1 "${@:2}"
