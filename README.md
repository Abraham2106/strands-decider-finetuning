# Strands Decider: clasificador de transacciones Stellar

Piloto de fine-tuning para clasificar propuestas de pago como `allow`, `escalate` o `deny`. El flujo objetivo llama al juez únicamente cuando la clasificación final es `escalate`; las restricciones deterministas del pago siguen siendo una barrera independiente. Este repo prepara el entrenamiento y la evaluación; no firma ni envía transacciones y todavía no está integrado al gateway de ALAIA.

**Estado:** el usuario aportó resultados del primer piloto: 8/9 aciertos, 0 aprobaciones incorrectas y un caso que debía escalar pero fue denegado con confianza 0.994. Se verificó la consistencia de esos archivos; ese resultado no demuestra SOTA. El corpus v2 añade 864 casos balanceados y métricas de escalamiento. La nueva versión aún no se entrenó en GPU.

## Corpus ampliado v2

Ejecutar `python generate_synthetic.py`: genera 864 propuestas con 288 ejemplos por clase. Particiones congeladas por familias de formato y grupos contrafactuales: **504 train, 144 validation, 216 test**. El manifiesto versionado fija los hashes de los archivos generados; los JSON se recrean localmente en lugar de duplicarlos en Git.

Incluye ambigüedad de destinatario/monto, intención ausente, múltiples pagos, condiciones sin resolver, instrucciones no confiables, citas benignas de ataques, violaciones de límites, operaciones administrativas, activos no permitidos y grants de agentes. Son propuestas abstractas sintéticas, no pagos reales ni XDR. `oracleFacts` y las etiquetas permanecen fuera de la entrada del modelo.

```bash
python generate_synthetic.py
python train_stellar_decider.py --data-dir data/synthetic-v2 --check
```

Entrenar después de configurar el entorno CUDA documentado abajo:

```bash
PYTHONPATH="$PWD/upstream-strands:$PWD/upstream-strands/src" python train_stellar_decider.py --data-dir data/synthetic-v2 --output stellar-v2-run-001 --epochs 3
```

La evaluación añade `escalation_recall` y `missed_escalations`. La selección de época usa validación, priorizando menos aprobaciones incorrectas tras el enrutamiento, luego menos escalaciones omitidas y finalmente macro F1. No se selecciona con prueba.

Raven quedó registrado y la autorización OAuth completó. Sus herramientas no estuvieron disponibles en la conversación de generación: los datos **no están validados por Raven**. Se usó documentación oficial y un oráculo local de reglas. [Detalles y límites](data/synthetic-v2/README.md).

El notebook v1 mantiene su revisión histórica. El notebook `notebooks/stellar_decider_molab_v2.py` carga el generador v2 y el entrenador desde una revisión fija.

## Archivos

- `train_stellar_decider.py`: continuación explícita de LoRA y cabeza pointer publicada, comparación con el modelo base, selección de época por validación y evaluación final.
- `data/stellar-cases.json`: 51 estados públicos del corpus de ALAIA, con procedencia por archivo. Son 24 allow, 21 deny y 6 escalate.
- `notebooks/stellar_decider_molab.py`: notebook liviano; descarga una revisión fija de este repo y del código oficial Strands. No duplica el dataset ni el entrenador.
- `.github/workflows/check.yml`: sintaxis y comprobaciones de datos, métricas y enrutamiento en CPU. No instala modelos ni consume GPU.

## Comprobar sin GPU ni dependencias externas

Desde la raíz, con Python 3.10 o superior:

```bash
python train_stellar_decider.py --check
python -m py_compile train_stellar_decider.py notebooks/stellar_decider_molab.py
```

La prueba verifica IDs y estados duplicados, las particiones disjuntas, las métricas y el rechazo o escalado ante política ausente, baja confianza y valores inválidos. La partición fija tiene 33 casos de entrenamiento, 9 de validación y 9 de prueba. Plantillas similares se repiten entre particiones: no es una prueba de generalización a transacciones reales.

## Entrenar en molab

Abrir el notebook desde el enlace **Open in molab** de abajo. En molab seleccionar RTX Pro 6000, esperar que conecte el kernel y ejecutar las celdas. Pulsar **Train and compare the pilot**. La comprobación CUDA impide iniciar entrenamiento en CPU. El notebook instala el código oficial fijado, descarga el modelo público, entrena tres épocas y ofrece un ZIP con checkpoint, particiones, historial y reporte. Descargarlo antes de que termine la sesión; los archivos temporales no tienen persistencia garantizada.

[Open in molab](https://molab.marimo.io/github/Abraham2106/strands-decider-finetuning/blob/main/notebooks/stellar_decider_molab.py)

El notebook carga entrenador y datos desde `16b3bca513711cd637517a818c5de9a1ed80c046`. Para probar cambios posteriores, actualizar `_project_revision` a un commit que contenga ambos archivos. El notebook anterior importado en molab conserva su código embebido: este enlace abre la versión controlada por GitHub.

## Entrenar en Linux o WSL2 con NVIDIA

No es necesario contratar AWS. En un entorno Python aislado con PyTorch compatible con la GPU y el driver CUDA:

```bash
git clone https://github.com/strands-labs/strands-decider.git upstream-strands
git -C upstream-strands checkout ea2519129102c6482b6db4f900da0ae652399106
python -m pip install -e './upstream-strands[cuda,train]'
PYTHONPATH="$PWD/upstream-strands:$PWD/upstream-strands/src" python train_stellar_decider.py --output stellar-run-001 --epochs 3
```

El modelo inicial es `StrandsAgents/strands-decider-2B-qwen3.5-v1-2610`; su revisión exacta del Hub se registra al ejecutarlo. La carpeta de salida debe ser nueva. PyTorch/transformers y los kernels CUDA requieren una prueba real de inferencia y backward en el host: los checks CPU no la sustituyen.

## Contrato de enrutamiento

1. Decodificar/validar la propuesta con Stellar SDK y evaluar las restricciones deterministas.
2. Política que niega → `deny`; política ausente o inválida → `escalate`.
3. Clasificación `escalate`, salida inválida o confianza inferior al umbral → `escalate`.
4. Solamente `escalate` invoca el juez. Los demás resultados no requieren juez dentro del flujo objetivo.

`route()` implementa los pasos 2–3. El umbral experimental es 0.9 sobre la confianza normalizada de choice, no una garantía de calibración. El reporte de cobertura usa el clasificador aislado; no simula el gate determinista ni ejecuta un juez. La implementación vigente de `alaia-stellar-proof` todavía exige juez obligatorio y no cambia por crear este repo.

## Alcance del piloto

Se entrena una sola pregunta de clasificación global usando `expected` del corpus. Incluye política e intención; no reemplaza directamente el contrato Verify de cinco preguntas. No se inventan etiquetas para las otras preguntas ni se mezclan las confirmaciones que tienen etiquetas distintas. Se conserva el adapter y la cabeza pointer del checkpoint publicado. La ruta oficial `init_from` inspeccionada reemplaza la cabeza y congela el torso, por lo que aquí se utiliza continuación explícita.

Se mide el modelo base y el ajustado con el mismo prompt y holdout. La época se selecciona con validación, priorizando menos aprobaciones incorrectas y después macro F1. El reporte incluye matriz de confusión, accuracy, macro F1, false-allow, cobertura y llamadas potenciales al juez. Nueve casos de prueba no permiten establecer tasas de error fiables para pagos automáticos; el ajuste no tiene garantizada una mejora.

Para aspirar a SOTA faltan datos amplios revisados, separación por plantillas/cuentas, ataques adversariales, calibración independiente y comparación con modelos fuertes sobre el mismo benchmark.

## Procedencia

- [Strands Decider y código de entrenamiento](https://github.com/strands-labs/strands-decider), código fijado en `ea2519129102c6482b6db4f900da0ae652399106`.
- [Modelo inicial en Hugging Face](https://huggingface.co/StrandsAgents/strands-decider-2B-qwen3.5-v1-2610).
- [Corpus de ALAIA](https://github.com/Abraham2106/alaia-stellar-proof/tree/9a41de6d674608f21c6531b6c85ad9e71d632a88/corpus/generic), referencia `9a41de6d674608f21c6531b6c85ad9e71d632a88`.
- [Cómputo y límites de molab](https://docs.marimo.io/guides/molab/).

Este repo no concede una nueva licencia a los datos importados; conserva su procedencia. El código oficial Strands se obtiene por separado bajo su licencia Apache-2.0.
