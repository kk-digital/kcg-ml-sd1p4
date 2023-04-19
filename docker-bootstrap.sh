#!/bin/bash
set -e
source /venv/bin/activate
update-ca-certificates --fresh
export SSL_CERT_DIR=/etc/ssl/certs
python3 /root/repo/main.py
