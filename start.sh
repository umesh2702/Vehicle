#!/usr/bin/env bash
export FLASK_APP=api/predict_api.py
export FLASK_ENV=production
flask run --host=0.0.0.0 --port=$PORT