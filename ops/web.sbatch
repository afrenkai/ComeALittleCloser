#!/bin/bash
set -euo pipefail
printf '%s\n' 'Turing is computation-only. Run the web application locally; SSH tunnels and remote web services are disabled.' >&2
exit 1
