import numpy as np
import skfuzzy as fuzzy
from skfuzzy import control as ctrl
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import os

from fastapi.middleware.cors import CORSMiddleware

# -----------------------------------------------------
# PASO 1: Inicialización de FastAPI y Definición de Esquemas
# -----------------------------------------------------

app = FastAPI(
    title="Servicio de Clasificación por Lógica Difusa",
    description="API que clasifica la calidad de servicio basándose en Tiempo de Espera y Calidad de la Comida."
)

# -----------------------------------------------------
# CONFIGURACIÓN CORS FINAL PARA PRUEBA LOCAL/PRODUCCIÓN
# -----------------------------------------------------
origins = [
    # Permitir solicitudes desde la propia máquina (localhost)
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    
    # *** ESTO ES CRUCIAL PARA ABRIR EL ARCHIVO HTML DIRECTAMENTE ***
    "null",
    
    # Permitir todos los orígenes (para la prueba de CORS más permisiva)
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# -----------------------------------------------------

# Define el formato de datos que esperamos recibir en la petición POST
class ClasificacionRequest(BaseModel):
    """Esquema de las variables de entrada para la clasificación."""
    tiempo_espera: float
    calidad_comida: float


# -----------------------------------------------------
# PASO 2: Definición del Sistema de Inferencia Difusa (FIS)
# -----------------------------------------------------

# Universo de Discurso
tiempo_espera = ctrl.Antecedent(np.arange(0, 61, 1), 'tiempo_espera')
calidad_comida = ctrl.Antecedent(np.arange(0, 11, 1), 'calidad_comida')
calidad_servicio = ctrl.Consequent(np.arange(0, 101, 1), 'calidad_servicio')

# Funciones de Pertenencia (Fuzzificación)
tiempo_espera['Bajo'] = fuzzy.trimf(tiempo_espera.universe, [0, 0, 15])
tiempo_espera['Medio'] = fuzzy.trimf(tiempo_espera.universe, [10, 25, 40])
# Se mantiene tu ajuste: empieza a ser Alto a los 20 min
tiempo_espera['Alto'] = fuzzy.trimf(tiempo_espera.universe, [20, 60, 60]) 

calidad_comida['Mala'] = fuzzy.trimf(calidad_comida.universe, [0, 0, 4])
calidad_comida['Aceptable'] = fuzzy.trimf(calidad_comida.universe, [3, 6, 8])
calidad_comida['Excelente'] = fuzzy.trimf(calidad_comida.universe, [7, 10, 10])

calidad_servicio['Mala'] = fuzzy.trimf(calidad_servicio.universe, [0, 0, 30])
calidad_servicio['Regular'] = fuzzy.trimf(calidad_servicio.universe, [20, 45, 70])
calidad_servicio['Buena'] = fuzzy.trimf(calidad_servicio.universe, [60, 80, 95])
calidad_servicio['Excelente'] = fuzzy.trimf(calidad_servicio.universe, [90, 100, 100])

# Definición de las 9 Reglas Lógicas
rule1 = ctrl.Rule(tiempo_espera['Bajo'] & calidad_comida['Mala'], calidad_servicio['Regular'])
rule2 = ctrl.Rule(tiempo_espera['Bajo'] & calidad_comida['Aceptable'], calidad_servicio['Buena'])
rule3 = ctrl.Rule(tiempo_espera['Bajo'] & calidad_comida['Excelente'], calidad_servicio['Excelente'])
rule4 = ctrl.Rule(tiempo_espera['Medio'] & calidad_comida['Mala'], calidad_servicio['Mala'])
rule5 = ctrl.Rule(tiempo_espera['Medio'] & calidad_comida['Aceptable'], calidad_servicio['Regular'])
rule6 = ctrl.Rule(tiempo_espera['Medio'] & calidad_comida['Excelente'], calidad_servicio['Buena'])
rule7 = ctrl.Rule(tiempo_espera['Alto'] & calidad_comida['Mala'], calidad_servicio['Mala'])
rule8 = ctrl.Rule(tiempo_espera['Alto'] & calidad_comida['Aceptable'], calidad_servicio['Regular'])
rule9 = ctrl.Rule(tiempo_espera['Alto'] & calidad_comida['Excelente'], calidad_servicio['Buena'])

# Creación y Simulación del Sistema de Control (Pre-computado para eficiencia)
control_sistema = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
clasificador_servicio = ctrl.ControlSystemSimulation(control_sistema)


# -----------------------------------------------------
# PASO 3: Definición del Endpoint (Ruta) de la API
# -----------------------------------------------------

@app.post("/clasificar")
def clasificar(request: ClasificacionRequest):
    """
    Recibe el tiempo de espera y la calidad de la comida.
    Devuelve la puntuación de calidad de servicio clasificada (0-100).
    """
    try:
        # Asignar inputs al sistema de simulación
        clasificador_servicio.input['tiempo_espera'] = request.tiempo_espera
        clasificador_servicio.input['calidad_comida'] = request.calidad_comida

        # Ejecutar el sistema de inferencia difusa (Defuzzificación)
        clasificador_servicio.compute()

        # Obtener y redondear el resultado final nítido (crisp)
        puntuacion_final = clasificador_servicio.output['calidad_servicio']

        # Devolver el resultado en formato JSON
        return {
            "puntuacion_final": round(puntuacion_final, 2),
            "estado": "Clasificación exitosa"
        }
    except Exception as e:
        # Manejo de errores (ej. si los valores están fuera del rango definido)
        return {
            "puntuacion_final": None,
            "estado": f"Error en la clasificación: {str(e)}"
        }

# -----------------------------------------------------
# PASO 4: Configuración de Ejecución Local (Opcional, para pruebas)
# -----------------------------------------------------

# Esta sección es solo para correrlo localmente
if __name__ == "__main__":
    # Azure usará un servidor web (Gunicorn) para llamar a 'app',
    # pero para pruebas locales, se usa uvicorn directamente.
    port = int(os.environ.get("PORT", 8000))
    print(f"Iniciando API localmente en el puerto {port}...")
    uvicorn.run("api_servicio:app", host="0.0.0.0", port=port, reload=True)