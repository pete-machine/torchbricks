#!/usr/bin/env bash
previous_tag=$(git tag --sort=-creatordate | sed -n 2p) # This requires that at least one tag have been created previously to work.
git shortlog "${previous_tag}.." | sed 's/^./    &/'
