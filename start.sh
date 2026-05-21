#!/bin/bash
uvicorn src.api.app:app --host 0.0.0.0 --port 10000 --workers 2