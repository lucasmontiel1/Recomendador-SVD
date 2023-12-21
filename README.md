## Descripción del Proyecto
Este proyecto implementa un servicio web utilizando FastAPI para recomendar productos a usuarios basándose en un modelo de filtrado colaborativo. El modelo utiliza descomposición de valor singular (SVD) para entrenarse en los datos de ventas y proporciona recomendaciones personalizadas.

## Estructura del Proyecto
El proyecto consta de los siguientes elementos:

Código Principal:

main.py: Contiene el código principal para el servidor FastAPI y la lógica del modelo de recomendación.

Datos:
ConsultaData.csv: Archivo de datos de ventas utilizado para entrenar el modelo.

Resultados y Modelos:
matrix.csv: Archivo que almacena la matriz de interacciones usuario-artículo después del preprocesamiento.
sparse_matrix.npz: Archivo que guarda la matriz dispersa utilizada en el modelo.


## Requisitos del Sistema
Asegúrate de tener las siguientes bibliotecas instaladas antes de ejecutar el código:
pip install uvicorn fastapi numpy pandas scipy scikit-learn pyodbc

## Configuración del Entorno
Configura las credenciales del servidor SQL en el script main.py para que se pueda establecer la conexión correctamente.
Asegúrate de tener permisos para acceder a la base de datos y leer las tablas necesarias.


## Instrucciones de Ejecución
Descarga de Datos:
Coloca el archivo de datos ConsultaData.csv en la ruta especificada en el script.
Configuración del Servidor:

Ajusta el host y el puerto en el script main.py según tus preferencias.

## Ejecución del Servidor:

Ejecuta el script recsysSVD.py para iniciar el servidor FastAPI.

## Entrenamiento del Modelo:

Llama a la ruta /prepararModelo mediante un navegador o herramienta API para entrenar el modelo.

## Consulta de Recomendaciones:

Accede a la ruta /consulta/{customer_id} para obtener recomendaciones para un usuario específico.


## Endpoints del Servicio
/health: Ruta para verificar el estado del servicio.
/metric: Ruta que devuelve la métrica de solicitudes al servidor.
/obtenerData: Ruta para obtener datos desde la base de datos.
/prepararModelo: Ruta para entrenar el modelo de recomendación.
/levantarModelo: Ruta para cargar el modelo previamente entrenado.
/consulta/{customer_id}: Ruta para obtener recomendaciones para un usuario específico.

