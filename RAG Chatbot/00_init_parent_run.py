# Databricks notebook source
from config.config_utils import start_parent_run

# Parametrización del entorno
try:
    environment = dbutils.widgets.get("environment")
except Exception:
    environment = "dev"

parent_run_id = start_parent_run(environment=environment)

# Exponer el parent_run_id como task value para el resto del pipeline
try:
    dbutils.jobs.taskValues.set(key="parent_run_id", value=parent_run_id)
except Exception as e:
    print(f"[WARNING] No se pudo exponer el parent_run_id como task value: {e}")

print(f"Parent run id: {parent_run_id}")