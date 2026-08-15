#!/usr/bin/env python3
"""
Scan a team state JSON file and sum `prompt_tokens` and `completion_tokens`.

Usage:
  python3 calculate.py /path/to/team_state_admin_...json

Prints overall totals and a breakdown by `source` when available.
"""
import argparse
import json
import sys
from collections import defaultdict


def safe_int(v):
	try:
		return int(v)
	except Exception:
		return 0


def traverse(obj, context_source, totals, per_source):
	if isinstance(obj, dict):
		# update source context if present
		source = obj.get("source", context_source)

		# check for models_usage dict
		mu = obj.get("models_usage")
		if isinstance(mu, dict):
			p = safe_int(mu.get("prompt_tokens"))
			c = safe_int(mu.get("completion_tokens"))
			totals["prompt_tokens"] += p
			totals["completion_tokens"] += c
			per_source[source]["prompt_tokens"] += p
			per_source[source]["completion_tokens"] += c

		# recurse into values
		for v in obj.values():
			traverse(v, source, totals, per_source)

	elif isinstance(obj, list):
		for item in obj:
			traverse(item, context_source, totals, per_source)


def main():
	p = argparse.ArgumentParser(description="Sum prompt/completion tokens in a team state JSON")
	p.add_argument("json_file", help="Path to the team state JSON file")
	args = p.parse_args()

	try:
		with open(args.json_file, "r", encoding="utf-8") as f:
			data = json.load(f)
	except Exception as e:
		print(f"Failed to open/parse JSON: {e}", file=sys.stderr)
		sys.exit(2)

	totals = {"prompt_tokens": 0, "completion_tokens": 0}
	per_source = defaultdict(lambda: {"prompt_tokens": 0, "completion_tokens": 0})

	traverse(data, None, totals, per_source)

	print("Overall totals:")
	print(f"  prompt_tokens: {totals['prompt_tokens']}")
	print(f"  completion_tokens: {totals['completion_tokens']}")
	print()
	print("Breakdown by source (None means no source in ancestor path):")
	for src, vals in sorted(per_source.items(), key=lambda kv: (kv[0] is None, str(kv[0]))):
		print(f"- {src}: prompt_tokens={vals['prompt_tokens']}, completion_tokens={vals['completion_tokens']}")


if __name__ == "__main__":
	main()
