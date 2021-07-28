#!/bin/sh

TEMPFILE=$(mktemp).pdf
dot -Tpdf -o "${TEMPFILE}" "$1"
xdg-open "${TEMPFILE}"

