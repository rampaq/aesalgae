#!/bin/sh

# install as edit to have packages in place
# to have compiled libraries location at the package location
# remember to recompile COdeRedLLL if needed for larger lattices
#find ~/packages -mindepth 1 -maxdepth 1 -type d -exec pip3 install -e {} +
find . -mindepth 1 -maxdepth 1 -type d -exec pip3 install -e {} \;
