# Trabajo en este repo

- El objetivo es clasificar propuestas Stellar como allow/escalate/deny; solo escalate llama al juez.
- Mantener la política determinista fuera del modelo. No firmar ni enviar pagos desde entrenamiento, evaluación o CI.
- Antes de cambiar código: leer README y el entrenador. Ejecutar `python train_stellar_decider.py --check` y compilar los archivos Python.
- Registrar revisión del código, modelo, datos, particiones e hiperparámetros de cada experimento.
- Comparar modelo base y ajustado sobre los mismos datos de prueba; seleccionar época/umbral únicamente con validación.
- No presentar precisión, calibración, mejora o SOTA sin mediciones. Los 51 casos actuales son un piloto sintético.
- Al modificar el entrenador o los datos, actualizar la revisión fijada del notebook en un commit posterior.
- Guardar checkpoints/resultados fuera de Git; no introducir secretos ni datos privados en este repo público.
