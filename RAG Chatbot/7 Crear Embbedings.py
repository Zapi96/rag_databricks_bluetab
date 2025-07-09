# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook de Generación de Embeddings con Model Serving Endpoint
# MAGIC
# MAGIC **Versión Final**
# MAGIC
# MAGIC Este notebook realiza el siguiente proceso:
# MAGIC 1.  Se conecta al Model Registry para obtener y descargar las dependencias (`requirements.txt`) de un modelo específico.
# MAGIC 2.  Instala esas dependencias en el entorno del notebook.
# MAGIC 3.  Carga una tabla de origen con datos de texto (`chunks`).
# MAGIC 4.  Llama a un **Model Serving Endpoint** (donde el modelo está desplegado) para generar los embeddings para cada texto en lotes.
# MAGIC 5.  Guarda los resultados, incluyendo los nuevos embeddings, en una tabla Delta final.
# MAGIC 6.  Esta tabla final está lista para ser utilizada en la creación de un índice de Databricks Vector Search.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Celda 1: Configuración Principal
# MAGIC
# MAGIC Define todas las variables necesarias. **Por favor, modifica estos valores.**

# COMMAND ----------

# ---- Nombres de Catálogo, Esquema y Tablas ----
CATALOG_NAME = "bluetab"
SCHEMA_NAME = "rag"
SOURCE_TABLE_NAME = "docs_text"      # Tu tabla con las columnas SID y chunk
DESTINATION_TABLE_NAME = "docs_text_embeddings" # La tabla final que crearemos

# ---- Configuración del Modelo y Endpoint ----

# URL del Model Serving Endpoint donde tienes el modelo desplegado.
# Ejemplo: "https://<tu-instancia>.cloud.databricks.com/serving-endpoints/<nombre-del-endpoint>/invocations"
ENDPOINT_URL = 'https://dbc-ad7d5e59-0280.cloud.databricks.com/serving-endpoints/simple_embbeding/invocations'


# ---- Construcción de nombres completos (No necesitas modificar esto) ----
source_table_fullname = f"{CATALOG_NAME}.{SCHEMA_NAME}.{SOURCE_TABLE_NAME}"
destination_table_fullname = f"{CATALOG_NAME}.{SCHEMA_NAME}.{DESTINATION_TABLE_NAME}"

print("Configuración cargada.")
print(f"Tabla de origen: {source_table_fullname}")
print(f"Tabla de destino: {destination_table_fullname}")
print(f"URL de Endpoint: {ENDPOINT_URL}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Celda 3: Cargar Librerías y Configurar Conexión al Endpoint

# COMMAND ----------

import requests
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.types import ArrayType, FloatType

# Cargamos el token de acceso desde Databricks Secrets
DATABRICKS_TOKEN = ""

print("Librerías importadas y token de acceso cargado.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Celda 4: Cargar la Tabla de Origen

# COMMAND ----------

try:
  source_df = spark.read.table(source_table_fullname)
  print(f"Tabla '{source_table_fullname}' cargada exitosamente.")
  print(f"Número de filas a procesar: {source_df.count()}")
  display(source_df.limit(5))
except Exception as e:
  print(f"Error al cargar la tabla '{source_table_fullname}'. Verifica que el nombre y los permisos son correctos.")
  raise e

# COMMAND ----------

# MAGIC %md
# MAGIC ### Celda 5: Definir Pandas UDF para Llamar al Endpoint en Lotes
# MAGIC
# MAGIC Esta es la función principal que se aplicará a los datos de Spark. Opera en lotes para mayor eficiencia.

# COMMAND ----------

import json

def get_embeddings_from_endpoint(text_series: pd.Series) -> pd.Series:
    """
    Toma una serie de pandas con textos, llama al endpoint en un solo POST y devuelve una serie con los embeddings.
    """
    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json",
    }
    data = {"inputs": text_series.tolist()}
    
    try:
        response = requests.post(ENDPOINT_URL, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        predictions = response.json().get("predictions", [])
        return pd.Series(predictions)
    except requests.exceptions.RequestException as e:
        print(f"Error en la llamada a la API: {e}")
        return pd.Series([None] * len(text_series))
    
# Registrar la función como una Pandas UDF
embeddings_from_endpoint_udf = F.pandas_udf(get_embeddings_from_endpoint, returnType=ArrayType(FloatType()))

print("Pandas UDF para llamar al endpoint creada exitosamente.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Celda 6: Aplicar la UDF y Generar los Embeddings

# COMMAND ----------

print("Aplicando la UDF al DataFrame para generar embeddings desde el endpoint...")
print("Este proceso puede tardar.")

# Aplicar la Pandas UDF a la columna 'chunk'
df_with_embeddings = source_df.withColumn(
    "embedding",
    embeddings_from_endpoint_udf(F.col("text"))
)

# Filtrar filas donde la llamada a la API pudo haber fallado
successful_embeddings_df = df_with_embeddings.filter(F.col("embedding").isNotNull())
failed_rows_count = df_with_embeddings.filter(F.col("embedding").isNull()).count()

print(f"Embeddings generados. Número de filas exitosas: {successful_embeddings_df.count()}")
if failed_rows_count > 0:
    print(f"ADVERTENCIA: Hubo {failed_rows_count} filas que no pudieron ser procesadas.")

display(successful_embeddings_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Celda 7: Guardar la Tabla Resultante
# MAGIC
# MAGIC Guarda el DataFrame con los embeddings en una nueva tabla Delta, lista para Vector Search.

# COMMAND ----------

successful_embeddings_df.printSchema()

# COMMAND ----------

from pyspark.sql.functions import expr

print(f"Guardando la tabla con embeddings en: '{destination_table_fullname}'...")

successful_embeddings_df = successful_embeddings_df.withColumn(
    "embedding",
    expr("transform(embedding, x -> cast(x as float))")
)
successful_embeddings_df.write.format("delta").mode("overwrite").saveAsTable(destination_table_fullname)

print("¡Proceso completado exitosamente!")

# COMMAND ----------

successful_embeddings_df.printSchema()

# COMMAND ----------

successful_embeddings_df.selectExpr("typeof(embedding[0]) as tipo_elemento").distinct().show()

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE `bluetab`.`rag`.`docs_text_embeddings` SET TBLPROPERTIES (delta.enableChangeDataFeed = true)

# COMMAND ----------

from pyspark.sql.functions import size, col

dimension_row = successful_embeddings_df.select(size(col("embedding")).alias("embedding_dimension")).first()

if dimension_row:
  embedding_dimension = dimension_row["embedding_dimension"]
  print(f"La dimensión del embedding es: {embedding_dimension}")
else:
  print("No se pudo determinar la dimensión. ¿La tabla está vacía?")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Siguientes Pasos: Crear el Índice de Vector Search
# MAGIC
# MAGIC Ahora que tienes tu tabla `tabla_con_embeddings` con la columna `embedding` pre-calculada, sigue estos pasos en la UI de Databricks:
# MAGIC
# MAGIC 1.  En el menú de la izquierda, ve a **Compute** (Cálculo).
# MAGIC 2.  Busca y haz clic en la pestaña **Vector Search**.
# MAGIC 3.  Haz clic en el botón **Create Index** (Crear Índice).
# MAGIC 4.  **Rellena el formulario:**
# MAGIC     * **Vector Search Index Name**: Dale un nombre a tu índice (ej: `mi_indice_chunks`).
# MAGIC     * **Endpoint**: Elige tu endpoint de Vector Search.
# MAGIC     * **Source Table**:
# MAGIC         * **Source Type**: Elige **Delta Table**.
# MAGIC         * **Source Table**: Busca y selecciona la tabla que acabas de crear (`DESTINATION_TABLE_NAME`).
# MAGIC         * **Primary Key**: Selecciona tu columna `SID`.
# MAGIC     * **Embedding Source**: **¡Esta es la parte clave!**
# MAGIC         * Selecciona **Use existing embedding column** (Usar columna de embedding existente).
# MAGIC         * **Embedding Column**: En el desplegable, elige la columna `embedding`.
# MAGIC     * **Sync Mode**: Elige `Continuous` o `Triggered`.
# MAGIC 5.  Haz clic en **Create**.
