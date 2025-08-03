# Importar librerías requeridas
import mlflow
import os
from datetime import datetime
from pyspark.sql import SparkSession

def cleanup_active_runs():
    """Parar todas las runs de MLflow activas"""
    try:
        active_runs = mlflow.search_runs(filter_string="status = 'RUNNING'")
        for run_id in active_runs['run_id']:
            mlflow.end_run(run_id)
        print("✅ Runs activas limpiadas")
    except Exception as e:
        print(f"⚠️ Error limpiando runs activas: {e}")

def log_step(step_name, status="started", details=None):
    """Log pipeline step information"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = f"[{timestamp}] Step: {step_name} - Status: {status}"
    if details:
        message += f" - Details: {details}"
    print(message)

def start_parent_run(run_name=None, environment="dev"):
    """
    Inicia una parent run para todo el pipeline RAG
    
    Args:
        run_name: Nombre opcional para la run
        environment: Ambiente (dev/test/prod)
    """
    if run_name is None:
        run_name = f"RAG_Pipeline_{environment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        # Asegurar que no hay runs activas
        cleanup_active_runs()
        
        # Iniciar parent run
        current_run = mlflow.start_run(run_name=run_name)
        parent_run_id = current_run.info.run_id
        
        # Log parámetros globales del pipeline
        mlflow.log_param("pipeline_type", "RAG_Databricks_Bluetab")
        mlflow.log_param("environment", environment)
        mlflow.log_param("timestamp", datetime.now().isoformat())
        mlflow.log_param("run_type", "parent")
        
        print(f"🚀 Parent run iniciada: {run_name}")
        print(f"📍 Run ID: {parent_run_id}")
        
        return parent_run_id
        
    except Exception as e:
        print(f"❌ Error iniciando parent run: {e}")
        raise e

def start_child_run(task_name, parent_run_id=None, environment="dev"):
    """
    Inicia una child run para una tarea específica del pipeline
    
    Args:
        task_name: Nombre de la tarea (ej: "01_create_tables", "02_pdf_processing")
        parent_run_id: ID de la parent run
        environment: Ambiente (dev/test/prod)
    """
    if parent_run_id is None:
        print("⚠️ No hay parent run activa. Iniciando parent run automáticamente...")
        parent_run_id = start_parent_run(environment=environment)
    
    try:
        # Iniciar child run
        run_name = f"{task_name}_{environment}_{datetime.now().strftime('%H%M%S')}"
        current_run = mlflow.start_run(
            run_name=run_name,
            nested=True
        )
        
        # Log parámetros de la child run
        mlflow.log_param("task_name", task_name)
        mlflow.log_param("parent_run_id", parent_run_id)
        mlflow.log_param("environment", environment)
        mlflow.log_param("run_type", "child")
        mlflow.log_param("timestamp", datetime.now().isoformat())
        
        print(f"📋 Child run iniciada: {run_name}")
        print(f"🔗 Parent run: {parent_run_id}")
        print(f"📍 Child run ID: {current_run.info.run_id}")
        
        log_step(task_name, "started", f"Child run iniciada para tarea {task_name}")
        
        return current_run.info.run_id
        
    except Exception as e:
        print(f"❌ Error iniciando child run: {e}")
        raise e

def end_child_run(status="success"):
    """
    Finaliza la child run actual
    """
    try:
        mlflow.log_param("final_status", status)
        mlflow.end_run()
        print(f"✅ Child run finalizada con status: {status}")
        
    except Exception as e:
        print(f"⚠️ Error finalizando child run: {e}")

def end_parent_run(status="success"):
    """
    Finaliza la parent run del pipeline
    """
    try:
        mlflow.log_param("pipeline_final_status", status)
        mlflow.log_param("pipeline_end_time", datetime.now().isoformat())
        mlflow.end_run()
        print(f"🏁 Parent run finalizada con status: {status}")
        
    except Exception as e:
        print(f"⚠️ Error finalizando parent run: {e}")

def get_current_run_info():
    """
    Obtiene información de las runs actuales
    """
    try:
        active_run = mlflow.active_run()
        if active_run:
            return {
                "current_run_id": active_run.info.run_id,
                "run_name": active_run.info.run_name,
                "status": active_run.info.status
            }
        else:
            return {
                "current_run_id": None,
                "run_name": None,
                "status": None
            }
    except:
        return {
            "current_run_id": None,
            "run_name": None,
            "status": None
        }

def create_table_if_not_exists(table_name, schema_sql):
    """Create table if it doesn't exist"""
    try:
        spark = get_spark_session()
        if spark is None:
            log_step("create_table", "failed", f"Spark session not available for {table_name}")
            return False
            
        spark.sql(f"CREATE TABLE IF NOT EXISTS {table_name} {schema_sql}")
        log_step("create_table", "success", f"Table {table_name} created/verified")
        return True
    except Exception as e:
        log_step("create_table", "failed", f"Error creating {table_name}: {e}")
        return False

def get_spark_session():
    """
    Obtiene la sesión de Spark activa si está disponible
    
    Returns:
        SparkSession: La sesión de Spark activa o None si no está disponible
    """
    try:
        # Usar SparkSession.getActiveSession() para obtener la sesión activa
        
        spark_session = SparkSession.getActiveSession()
        
        if spark_session is not None:
            # Verificar que la sesión esté activa probando una operación simple
            try:
                # Probar una operación básica para verificar que funciona
                test_df = spark_session.range(1)
                test_df.count()
                return spark_session
            except Exception as e:
                print(f"⚠️ Sesión de Spark encontrada pero no funcional: {e}")
                return None
        return None
    except Exception as e:
        print(f"⚠️ Error obteniendo sesión de Spark: {e}")
        return None

def get_spark_safe():
    """
    Obtiene la sesión de Spark de forma segura para usar en notebooks
    
    Returns:
        SparkSession: La sesión de Spark activa
        
    Raises:
        Exception: Si no hay sesión de Spark disponible
    """
    spark = get_spark_session()
    if spark is None:
        raise Exception("No hay sesión de Spark disponible. Verifica que el cluster esté activo y el notebook esté conectado.")
    return spark

def get_spark_or_none():
    """
    Obtiene la sesión de Spark de forma segura, retorna None si no está disponible
    
    Returns:
        SparkSession or None: La sesión de Spark activa o None
    """
    return get_spark_session()

def detect_spark_session():
    """
    Detecta si hay una sesión de Spark activa y retorna información sobre ella
    
    Returns:
        dict: Información de la sesión de Spark o None si no está disponible
    """
    try:
        # Intentar acceder a la variable spark global
        if 'spark' in globals():
            spark_session = globals()['spark']
            if spark_session is not None:
                return {
                    "available": True,
                    "session_id": spark_session._jsc.sc().applicationId(),
                    "app_name": spark_session.conf.get("spark.app.name", "Unknown"),
                    "master": spark_session.conf.get("spark.master", "Unknown"),
                    "databricks_workspace": spark_session.conf.get("spark.databricks.workspaceUrl", "Unknown"),
                    "cluster_id": spark_session.conf.get("spark.databricks.clusterUsageTags.clusterId", "Unknown"),
                    "driver_node_type": spark_session.conf.get("spark.databricks.clusterUsageTags.driverNodeType", "Unknown"),
                    "worker_node_type": spark_session.conf.get("spark.databricks.clusterUsageTags.workerNodeType", "Unknown"),
                    "runtime_version": spark_session.conf.get("spark.databricks.clusterUsageTags.runtimeVersion", "Unknown")
                }
        return None
    except Exception as e:
        print(f"⚠️ Error detectando sesión de Spark: {e}")
        return None

def get_spark_info():
    """
    Obtiene información detallada de la sesión de Spark activa
    
    Returns:
        dict: Información completa de Spark o mensaje de error
    """
    spark_info = detect_spark_session()
    
    if spark_info is None:
        return {
            "status": "no_session",
            "message": "No hay sesión de Spark activa disponible",
            "available": False
        }
    
    try:
        # Obtener información adicional si está disponible
        spark = get_spark_session()
        if spark is None:
            return {
                "status": "error",
                "message": "No se pudo obtener sesión de Spark",
                "available": False
            }
        
        # Información del cluster
        cluster_info = {
            "num_executors": spark.conf.get("spark.executor.instances", "Unknown"),
            "executor_memory": spark.conf.get("spark.executor.memory", "Unknown"),
            "executor_cores": spark.conf.get("spark.executor.cores", "Unknown"),
            "driver_memory": spark.conf.get("spark.driver.memory", "Unknown")
        }
        
        # Combinar información
        full_info = {
            "status": "active",
            "available": True,
            "session_info": spark_info,
            "cluster_info": cluster_info,
            "message": "Sesión de Spark activa detectada"
        }
        
        return full_info
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error obteniendo información de Spark: {e}",
            "available": False
        }

def print_spark_status():
    """
    Imprime el estado actual de la sesión de Spark de forma legible
    """
    spark_info = get_spark_info()
    
    print("🔍 Estado de la Sesión de Spark:")
    print("=" * 50)
    
    if spark_info["available"]:
        session = spark_info["session_info"]
        cluster = spark_info["cluster_info"]
        
        print(f"✅ Estado: {spark_info['status']}")
        print(f"📱 Session ID: {session['session_id']}")
        print(f"🏷️ App Name: {session['app_name']}")
        print(f"🌐 Workspace: {session['databricks_workspace']}")
        print(f"🆔 Cluster ID: {session['cluster_id']}")
        print(f"💻 Driver Node: {session['driver_node_type']}")
        print(f"⚙️ Worker Node: {session['worker_node_type']}")
        print(f"📦 Runtime: {session['runtime_version']}")
        print(f"👥 Executors: {cluster['num_executors']}")
        print(f"💾 Executor Memory: {cluster['executor_memory']}")
        print(f"🔧 Executor Cores: {cluster['executor_cores']}")
        print(f"🚀 Driver Memory: {cluster['driver_memory']}")
    else:
        print(f"❌ {spark_info['message']}")
    
    print("=" * 50)

def validate_spark_session():
    """
    Valida que la sesión de Spark esté disponible y funcional
    
    Returns:
        bool: True si la sesión está disponible y funcional
    """
    try:
        spark_info = get_spark_info()
        
        if not spark_info["available"]:
            print(f"❌ Sesión de Spark no disponible: {spark_info['message']}")
            return False
        
        # Probar una operación simple para verificar que funciona
        spark = get_spark_session()
        if spark is None:
            print("❌ No se pudo obtener sesión de Spark")
            return False
            
        test_df = spark.range(1)
        test_df.count()
        
        print("✅ Sesión de Spark validada y funcional")
        return True
        
    except Exception as e:
        print(f"❌ Error validando sesión de Spark: {e}")
        return False

def get_databricks_host():
    """Get Databricks workspace URL"""
    try:
        spark = get_spark_session()
        if spark is None:
            return "https://dbc-ad7d5e59-0280.cloud.databricks.com/"
        return spark.conf.get("spark.databricks.workspaceUrl")
    except:
        return "https://dbc-ad7d5e59-0280.cloud.databricks.com/"

def build_endpoint_url(endpoint_name):
    """Build serving endpoint URL"""
    host = get_databricks_host()
    return f"https://{host}/serving-endpoints/{endpoint_name}/invocations"

def setup_mlflow_experiment(experiment_name="/Shared/RAG_Databricks_Bluetab_Pipeline"):
    """Configurar MLflow experiment"""
    mlflow.set_registry_uri("databricks-uc")
    
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f"Creado nuevo experimento: {experiment_name}")
        else:
            experiment_id = experiment.experiment_id
            print(f"Usando experimento existente: {experiment_name}")
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        print(f"Error configurando experimento: {e}")
        mlflow.set_experiment("/Shared/RAG_Default") 