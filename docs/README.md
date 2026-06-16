# Manual de usuario: pruebas del baseline y del modelo propuesto

Este documento explica como ejecutar localmente el modelo base del repositorio y el modelo propuesto desarrollado en el TFG. El objetivo es poder reproducir una inferencia sobre un ensayo, obtener fonemas, generar texto con el modelo de lenguaje y comparar ambos modelos.

## 1. Estructura utilizada

Las pruebas se ejecutan desde el repositorio original:

```bash
cd /home/raquel/Escritorio/TFG_Brain-to-text/nejm-brain-to-text-main
```

Los componentes principales son:

| Componente | Ubicacion | Funcion |
|---|---|---|
| Baseline RNN-GRU | `data/t15_pretrained_rnn_baseline` | Modelo neuronal preentrenado proporcionado por el repositorio. |
| Modelo propuesto | `mi_modelo/` | Variante basada en el baseline con adaptador residual y corrector temporal de logits. |
| Datos HDF5 | `data/hdf5_data_final` | Ensayos de entrenamiento, validacion y test. |
| Modelo de lenguaje | `language_model/` | Decodificador que transforma logits foneticos en texto. |
| Scripts de prueba | `run_single_trial_text_decode.py` y `mi_modelo/*.py` | Ejecucion y comparacion de modelos. |

## 2. Requisitos previos

Antes de ejecutar las pruebas deben estar disponibles:

- Entorno Python del proyecto neuronal: `.venv-linux`.
- Entorno Conda del modelo de lenguaje: `b2txt25_lm`.
- Redis instalado.
- Modelo de lenguaje 1-gram en `language_model/pretrained_language_models/openwebtext_1gram_lm_sil`.
- Checkpoint del modelo propuesto en `mi_modelo/salidas/baseline_adapter_logit/best_checkpoint.pt`.

Si ya se han ejecutado las pruebas anteriormente, normalmente solo hay que levantar Redis y el modelo de lenguaje.

## 3. Descarga y preparacion de los datos

Los datos originales se descargan desde Dryad:

```text
https://datadryad.org/stash/dataset/doi:10.5061/dryad.dncjsxm85
```

Una vez descargados, deben colocarse dentro de la carpeta `data` del repositorio:

```text
nejm-brain-to-text-main/data/
```

Antes de ejecutar el codigo hay que descomprimir, como minimo, los siguientes archivos:

```text
t15_copyTask_neuralData.zip
t15_pretrained_rnn_baseline.zip
```

La estructura esperada es:

```text
data/
├── hdf5_data_final/
│   ├── t15.2023.08.11/
│   ├── t15.2023.08.13/
│   └── ...
├── t15_pretrained_rnn_baseline/
│   └── checkpoint/
└── t15_copyTaskData_description.csv
```

El archivo `t15_copyTaskData_description.csv` contiene una descripcion bloque a bloque de los datos de la tarea Copy Task. Incluye informacion como la fecha de sesion, el numero de bloque, el numero de ensayos, el corpus utilizado y la particion a la que pertenece cada bloque (`train`, `val` o `test`). Este archivo no es obligatorio para ejecutar el codigo, pero resulta util para entender la organizacion del dataset.

## 4. Terminales necesarias

Para obtener texto final se necesitan tres terminales abiertas al mismo tiempo.

### Terminal 1: Redis

Desde cualquier ruta:

```bash
redis-server
```

Si aparece un mensaje indicando que el puerto `6379` ya esta en uso, significa que Redis ya esta ejecutandose. En ese caso no hace falta abrir otro servidor.

### Terminal 2: modelo de lenguaje

Desde la raiz del repositorio `nejm-brain-to-text-main`:

```bash
conda activate b2txt25_lm

python language_model/language-model-standalone.py \
  --lm_path language_model/pretrained_language_models/openwebtext_1gram_lm_sil \
  --nbest 10 \
  --acoustic_scale 0.325 \
  --blank_penalty 90 \
  --alpha 0.55 \
  --redis_ip localhost \
  --gpu_number -1
```

Este proceso queda esperando logits mediante Redis. Hay que dejar esta terminal abierta mientras se ejecutan las pruebas.

### Terminal 3: ejecucion de modelos

Desde la raiz del repositorio `nejm-brain-to-text-main` se ejecutan los scripts del baseline y del modelo propuesto.

## 5. Probar el baseline preentrenado

Para ejecutar un ensayo de validacion con el modelo base:

```bash
.venv-linux/bin/python run_single_trial_text_decode.py \
  --split val \
  --session_index 1 \
  --trial_index 0
```

El script muestra:

- sesion utilizada,
- particion del dataset,
- identificador del trial,
- forma de `input_features` tras suavizado,
- forma de los logits de la RNN,
- fonemas predichos por la RNN,
- frase real y fonemas reales si el ensayo pertenece a validacion,
- texto parcial del modelo de lenguaje,
- mejor frase final,
- lista de candidatos generados.

Ejemplo de uso con test:

```bash
.venv-linux/bin/python run_single_trial_text_decode.py \
  --split test \
  --session_index 1 \
  --trial_index 0
```

En test no se muestran frase real ni fonemas reales porque esta particion no contiene referencias accesibles localmente. Por tanto, en test no se calculan PER ni WER.

## 6. Probar el modelo propuesto

Para ejecutar el mismo ensayo con el modelo propuesto:

```bash
.venv-linux/bin/python mi_modelo/run_adapter_trial_text_decode.py \
  --split val \
  --session_index 1 \
  --trial_index 0
```

Este script realiza el mismo flujo que el baseline, pero usando el checkpoint del modelo propuesto:

```text
mi_modelo/salidas/baseline_adapter_logit/best_checkpoint.pt
```

La salida incluye:

- fonemas generados por el modelo propuesto,
- frase real si el ensayo es de validacion,
- candidatos generados por el modelo de lenguaje,
- PER de validacion guardado en el checkpoint.

Para probar un ensayo de test:

```bash
.venv-linux/bin/python mi_modelo/run_adapter_trial_text_decode.py \
  --split test \
  --session_index 1 \
  --trial_index 0
```

De nuevo, en test solo se obtiene prediccion, no metrica local.

## 7. Comparar baseline y modelo propuesto

Para comparar ambos modelos sobre varios ensayos de validacion:

```bash
.venv-linux/bin/python mi_modelo/compare_trials_text.py \
  --split val \
  --session_index 1 \
  --start_trial 0 \
  --num_trials 10
```

En validacion, el script puede calcular:

- PER del baseline,
- PER del modelo propuesto,
- WER del baseline,
- WER del modelo propuesto,
- mejor frase generada por cada modelo,
- si la frase final cambia o no.

Para comparar ensayos de test:

```bash
.venv-linux/bin/python mi_modelo/compare_trials_text.py \
  --split test \
  --session_index 1 \
  --start_trial 0 \
  --num_trials 10
```

En test la comparacion es cualitativa. El script muestra las frases generadas por ambos modelos, pero no calcula PER ni WER porque no existen referencias reales.

## 8. Evaluar PER del modelo propuesto

Para evaluar el modelo propuesto sobre validacion:

```bash
.venv-linux/bin/python mi_modelo/eval_per.py
```

Esta prueba se centra en la parte neuronal del sistema. Calcula el error fonetico comparando los fonemas predichos con los fonemas reales del conjunto de validacion.

## 9. Entrenar el ajuste fino del modelo propuesto

El entrenamiento completo del baseline original requiere hardware con GPU de alto rendimiento. En este trabajo se realiza ajuste fino sobre el modelo preentrenado.

Para entrenar la propuesta final desde el baseline preentrenado:

```bash
.venv-linux/bin/python mi_modelo/train_baseline_adapter.py \
  --logit_adapter \
  --output_dir mi_modelo/salidas/baseline_adapter_logit
```

Este comando carga el checkpoint original del baseline, congela las partes indicadas por la configuracion del script y entrena los modulos anadidos. Por defecto se trabaja con la sesion `t15.2023.08.13`.

Si se quiere lanzar una prueba reducida:

```bash
.venv-linux/bin/python mi_modelo/train_reducido.py \
  --max_train_trials 128 \
  --max_val_trials 64 \
  --batch_size 4 \
  --num_batches 100 \
  --eval_every 25
```

El entrenamiento guarda el mejor checkpoint cuando mejora el PER de validacion.

## 10. Interpretacion de resultados

Es importante diferenciar validacion y test:

| Particion | Tiene referencias reales | Uso principal | Permite PER/WER local |
|---|---|---|---|
| Train | Si | Entrenamiento | Si, aunque no se usa como comparacion final |
| Val | Si | Evaluacion y comparacion de modelos | Si |
| Test | No | Generar predicciones finales | No |

Por tanto:

- Los valores de PER y WER del trabajo se calculan sobre validacion.
- Los ensayos de test se usan solo para observar predicciones.
- Si aparece PER en una tabla, debe corresponder a validacion, no a test.

## 11. Problemas frecuentes

### Redis indica que el puerto esta ocupado

Significa que ya hay un servidor Redis activo. No hace falta abrir otro.

### El script se queda esperando al modelo de lenguaje

Comprueba que la Terminal 2 sigue ejecutando `language-model-standalone.py`. Si se ha cerrado, vuelve a iniciarla.

### `ModuleNotFoundError: torch` o `ModuleNotFoundError: transformers`

El entorno activo no es el correcto o faltan dependencias. Para el modelo de lenguaje debe estar activo:

```bash
conda activate b2txt25_lm
```

Para los scripts neuronales se usa:

```bash
.venv-linux/bin/python ...
```

### No aparece frase real en test

Es el comportamiento esperado. Los archivos de test contienen la señal neuronal, pero no incluyen frase real ni secuencia fonetica de referencia.
