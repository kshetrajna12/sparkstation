#!/usr/bin/env bash
# Push monitoring/grafana-dashboard-fixed.json to Grafana via API.
# Uses GRAFANA_URL + GRAFANA_API_TOKEN from the repo .env (gitignored).
# The service account (sparkstation-sa, Editor) needs Edit permission on the
# dashboard — granted via dashboard Settings → Permissions (2026-08-15).
set -euo pipefail
cd "$(dirname "$0")/.."
TOK=$(grep '^GRAFANA_API_TOKEN=' .env | cut -d= -f2)
URL=$(grep '^GRAFANA_URL=' .env | cut -d= -f2)
python3 - "$URL" "$TOK" <<'PY'
import json, sys, urllib.request
url, tok = sys.argv[1], sys.argv[2]
d = json.load(open('monitoring/grafana-dashboard-fixed.json'))
d.pop('id', None)
body = json.dumps({"dashboard": d, "overwrite": True,
                   "message": "pushed from sparkstation repo"}).encode()
req = urllib.request.Request(f'{url}/api/dashboards/db', data=body,
    headers={'Authorization': f'Bearer {tok}', 'Content-Type': 'application/json'})
r = json.loads(urllib.request.urlopen(req).read())
print(f"✓ {r['status']}: {r['uid']} now at version {r['version']}")
PY
