# Mi modelo: GRU compacto con proyeccion previa

Esta carpeta contiene la propuesta propia del TFG. La idea no es sustituir todo el pipeline original, sino modificar la fase neuronal del baseline y mantener la misma salida fonetica de 41 clases.

## Idea del modelo

El baseline introduce en la GRU una ventana temporal concatenada de 7168 valores. El modelo propuesto anade una proyeccion aprendida antes de la GRU:

```text
input_features
  -> suavizado gaussiano
  -> adaptacion especifica por sesion
  -> Softsign + Dropout
  -> ventanas temporales
  -> proyeccion lineal 7168 -> 512
  -> LayerNorm + GELU + Dropout
  -> GRU reducida
  -> capa lineal
  -> logits foneticos
```

Comparacion conceptual:

| Componente | Baseline | Modelo propuesto |
|---|---:|---:|
| Entrada por ventana | 7168 | 7168 |
| Proyeccion previa | no | 7168 -> 512 |
| Capas GRU | 5 | 2 |
| Unidades GRU | 768 | 256 |
| Salida | 41 clases | 41 clases |

El objetivo es reducir coste computacional manteniendo una representacion aprendida antes de la GRU.

## 1. Probar que la arquitectura funciona

Desde la raiz del repositorio:

```bash
.venv-linux/bin/python mi_modelo/smoke_test.py
```

Esto no entrena; solo carga un ensayo real, aplica suavizado, pasa por el modelo propuesto y muestra las formas.

## 2. Comparar numero de parametros

```bash
.venv-linux/bin/python mi_modelo/comparar_parametros.py
```

Sirve para justificar que el modelo propuesto es mas ligero que el baseline.

## 3. Entrenamiento reducido

```bash
.venv-linux/bin/python mi_modelo/train_reducido.py
```

Por defecto entrena una version pequena con:

- una sesion,
- pocos ensayos,
- pocos batches,
- CPU.

Se puede aumentar un poco:

```bash
.venv-linux/bin/python mi_modelo/train_reducido.py \
  --max_train_trials 128 \
  --max_val_trials 64 \
  --batch_size 4 \
  --num_batches 50
```

El mejor checkpoint se guarda en:

```text
mi_modelo/salidas/gru_compacto/best_checkpoint.pt
```

Para continuar entrenando desde el mejor checkpoint guardado:

```bash
.venv-linux/bin/python mi_modelo/train_reducido.py \
  --resume_checkpoint mi_modelo/salidas/gru_compacto/best_checkpoint.pt \
  --max_train_trials 128 \
  --max_val_trials 64 \
  --batch_size 4 \
  --num_batches 300 \
  --lr 0.0003 \
  --lr_min 0.00001 \
  --eval_every 50
```

## 4. Evaluar PER en validacion

```bash
.venv-linux/bin/python mi_modelo/eval_per.py
```

Esta evaluacion calcula el PER del modelo propuesto sobre validacion. El resultado se puede comparar con el baseline preentrenado, cuyo checkpoint reporta:

```text
val_PER = 0.1010
```

## Texto corto para la memoria

El modelo propuesto modifica la fase neuronal del baseline. Se mantiene la adaptacion por sesion, la agrupacion temporal y la salida fonetica, pero se introduce una proyeccion aprendida antes de la GRU. Esta proyeccion reduce la dimensionalidad de las ventanas temporales antes del procesamiento recurrente, permitiendo utilizar una GRU mas pequena. El modelo de lenguaje se mantiene igual que en el baseline para que la comparacion se centre en la arquitectura neuronal.
