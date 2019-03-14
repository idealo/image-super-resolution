#!/usr/bin/env bash

cp ../README.md docs/index.md
cp ../CONTRIBUTING.md docs/CONTRIBUTING.md
cp ../LICENSE docs/LICENSE.md
mkdir ../docs
mkdocs build -c -d ../docs/