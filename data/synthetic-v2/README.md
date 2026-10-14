# Corpus sintético v2

864 propuestas abstractas, balanceadas: 288 allow, 288 escalate, 288 deny. No son transacciones observadas, envelopes XDR ni pagos enviados.

Particiones congeladas: train 504 (168/clase), validation 144 (48/clase), test 216 (72/clase). Los grupos contrafactuales y las familias de formatos del operador no se cruzan entre particiones. Las situaciones semánticas siguen compartidas; este conjunto prueba variantes sintéticas, no generalización completa a situaciones nuevas.

Generar desde la raíz: `python generate_synthetic.py`. Validar: `python train_stellar_decider.py --data-dir data/synthetic-v2 --check`.

## Supervisión

`state` es la única entrada al modelo. `expected`, `family`, `scenario`, `counterfactual_group` y `oracleFacts` son metadatos externos y no se envían al clasificador. El generador etiqueta con un oráculo de política e intención, no con predicciones del mismo modelo que se entrena.

El oráculo valida límites, tipo de operación, activo permitido, red autorizada y grants sintéticos. Luego comprueba la intención única, cambios de destinatario/monto e instrucciones en evidencia no confiable. La política denegatoria tiene precedencia. Se incluyen citas benignas de instrucciones maliciosas como controles negativos.

Las direcciones solo tienen formato/checksum StrKey correcto: no se comprueba existencia en el ledger ni validez del punto criptográfico. No se generan ni guardan secretos. Grants, restricciones de activo nativo y etiquetas ALAIA son reglas de aplicación; no se atribuyen al consenso Stellar.

## Raven y fuentes

Raven fue registrado mediante `codex mcp add stellar-raven --url https://raven.stellar.org/mcp` y OAuth terminó correctamente. Sus herramientas no estaban disponibles en esta conversación al generar el corpus. **No se presenta el dataset como validado o generado por Raven.** El manifiesto registra esta limitación y enlaces a documentación oficial.

Los 51 casos originales siguen en `data/stellar-cases.json` como regresión histórica. No se mezclan silenciosamente con las nuevas particiones. El caso `rejected-23` que ya se inspeccionó no constituye una prueba nueva independiente.

Próxima verificación: usar Raven para localizar documentación por escenario y contrastar el esquema con Stellar SDK, añadir comprobaciones XDR offline y revisión independiente de etiquetas. No publicar una afirmación SOTA a partir de este corpus.
