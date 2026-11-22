#!/bin/bash
# usage: mkvariant.sh <name> <profile> '<python dict literal>'  -> commits on branch v/<name>, prints sha
set -e
cd "$(dirname "$0")"
name=$1; prof=$2; ov=$3
base=$(git rev-parse --abbrev-ref HEAD)
git checkout -q -B "v/$name"
sed -i '' -E "s/^ACTIVE_PROFILE = .*/ACTIVE_PROFILE = \"$prof\"/; s/^OVERRIDES = .*/OVERRIDES = $ov/" dex_bot.py
python3 kit/check/check.py --map-dir kit/arena -- python3 -u dex_bot.py 2>&1 | tail -1 >&2
git commit -qam "variant $name: profile=$prof overrides=$ov"
git push -q -f origin "v/$name" 2>/dev/null
git rev-parse HEAD
git checkout -q "$base"
