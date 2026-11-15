# Guía de replicación y publicación del proyecto

**Proyecto:** Predicción del riesgo de enfermedad coronaria a 10 años con Regresión Logística (Framingham Heart Study)

Esta guía permite a cualquier integrante del equipo:

1. preparar su computadora;
2. ejecutar el notebook y comprobar que obtiene los mismos resultados;
3. ejecutar el dashboard en su máquina;
4. publicar el dashboard en internet (Streamlit Community Cloud) para la entrega.

Sigue los pasos **en orden** y marca cada casilla al terminarla.

---

## 0. Qué contiene el proyecto

```
FINAL/
├── Proyecto_Regresion_Logistica_CHD.ipynb   ← Notebook principal (entregable técnico)
├── requirements.txt                          ← Librerías del proyecto completo
├── GUIA_REPLICACION.md                       ← Esta guía
├── .gitignore                                ← Archivos que NO se suben a GitHub
├── .streamlit/
│   └── config.toml                           ← Colores y tema del dashboard
├── dashboard/
│   ├── app.py                                ← Código del dashboard
│   └── requirements.txt                      ← Librerías mínimas del dashboard (para la nube)
├── data/
│   ├── raw/framingham.csv                    ← Datos originales
│   └── processed/
│       ├── framingham_limpio.csv             ← Generado por el notebook
│       └── predicciones_prueba.csv           ← Generado por el notebook
├── models/
│   ├── modelo_regresion_logistica.pkl        ← Modelo entrenado (generado por el notebook)
│   └── metricas.json                         ← Métricas (generado por el notebook)
└── figures/                                  ← 17 gráficas PNG (generadas por el notebook)
```

**Flujo:** el **notebook** limpia los datos, entrena el modelo y guarda los archivos de `data/processed/`, `models/` y `figures/`. El **dashboard** solo **lee** esos archivos. Por eso el notebook siempre va primero.

> ⚠️ **No incluir en la entrega** el archivo `logistic-regression-heart-diseases-prediction.ipynb` (material de referencia), el `framingham.csv` suelto en la raíz (ya hay una copia en `data/raw/`), ni la carpeta `.claude/`. El `.gitignore` ya los excluye de GitHub.

---

## 1. Preparar la computadora

### 1.1 Instalar Python

- [ ] Descarga Python **3.12 o superior** desde https://www.python.org/downloads/
- [ ] **Windows:** en el instalador marca la casilla **"Add Python to PATH"** antes de dar clic en *Install*.
- [ ] Verifica la instalación en una terminal (PowerShell en Windows, Terminal en Mac):

```bash
python --version
```

Debe mostrar `Python 3.12.x` o superior. En Mac, si no funciona, usa `python3` en lugar de `python` en todos los comandos de esta guía.

### 1.2 Copiar la carpeta del proyecto

- [ ] Copia la carpeta completa del proyecto (`FINAL/`) a tu computadora, desde el Drive/OneDrive del equipo o desde el repositorio de GitHub (sección 4).
- [ ] Abre una terminal **dentro de esa carpeta**:
  - **Windows:** abre la carpeta en el Explorador, escribe `powershell` en la barra de direcciones y presiona Enter.
  - **Mac:** clic derecho sobre la carpeta → *Nuevo terminal en la carpeta*.

### 1.3 Crear un entorno virtual (recomendado)

Un entorno virtual mantiene las librerías del proyecto separadas del resto de tu computadora.

```bash
python -m venv .venv
```

Actívalo (hay que hacerlo **cada vez** que abras una terminal nueva):

- **Windows (PowerShell):**
  ```bash
  .venv\Scripts\Activate.ps1
  ```
  Si aparece un error de "ejecución de scripts deshabilitada", ejecuta una sola vez `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, acepta y vuelve a intentar.
- **Mac / Linux:**
  ```bash
  source .venv/bin/activate
  ```

- [ ] Al activarlo, el inicio de la línea de la terminal muestra `(.venv)`.

### 1.4 Instalar las librerías

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

- [ ] Tarda unos minutos. Al final debe aparecer `Successfully installed ...` sin errores en rojo.

> Las versiones están **fijadas** en `requirements.txt` a propósito: el modelo `.pkl` se guardó con `scikit-learn 1.9.1`, y abrirlo con otra versión puede fallar o dar advertencias.

---

## 2. Ejecutar el notebook

### 2.1 Abrir Jupyter

```bash
python -m jupyter notebook
```

Se abrirá el navegador. Da clic en **`Proyecto_Regresion_Logistica_CHD.ipynb`**.

> También puedes abrirlo con **VS Code** (extensión *Jupyter*), seleccionando como kernel el Python de `.venv`.

### 2.2 Ejecutar todo desde cero

- [ ] Menú **Kernel → Restart Kernel and Run All Cells** (o *Run → Run All Cells*).
- [ ] Espera a que termine (aprox. 1–2 minutos). Ninguna celda debe mostrar un error en rojo.

### 2.3 Verificar que los resultados coinciden

Como el notebook usa una semilla fija (`RANDOM_STATE = 0`), los resultados deben ser **idénticos** a estos:

| Dónde revisar | Valor esperado |
|---|---|
| Sección 7: tamaños | Entrenamiento: 3,390 · Prueba: 848 · 15.2 % con CHD en ambos |
| Sección 12: modelo base | Sensibilidad 0.078 · ROC-AUC 0.713 |
| Sección 14.4: GridSearchCV | Mejores hiperparámetros: `C = 1`, `penalty = 'l2'` |
| Sección 15: umbral | Umbral óptimo (máximo F2): **0.12** |
| Sección 16: modelo final (umbral 0.12) | Sensibilidad **0.798** · Especificidad 0.508 · Precisión 0.225 · ROC-AUC **0.713** |
| Sección 18: archivos | Se generan los 4 archivos de `models/` y `data/processed/` |

- [ ] Todos los valores coinciden. Si alguno es distinto, revisa que las versiones de las librerías sean las de `requirements.txt` (`python -m pip list`).

### 2.4 Exportar el notebook a HTML (para la entrega y para quien no tenga Jupyter)

Cierra Jupyter (Ctrl + C en la terminal) y ejecuta:

```bash
python -m jupyter nbconvert --to html Proyecto_Regresion_Logistica_CHD.ipynb
```

- [ ] Se crea `Proyecto_Regresion_Logistica_CHD.html`, que se abre con cualquier navegador.

---

## 3. Ejecutar el dashboard en tu computadora

Desde la carpeta raíz del proyecto (con el entorno virtual activado):

```bash
python -m streamlit run dashboard/app.py
```

- [ ] Se abre el navegador en `http://localhost:8501`. Si no se abre solo, copia esa dirección en el navegador.
- [ ] Para detenerlo: Ctrl + C en la terminal.

> El comando debe ejecutarse **desde la carpeta raíz** (`FINAL/`), no desde `dashboard/`, para que se aplique el tema de `.streamlit/config.toml`.

### 3.1 Lista de pruebas del dashboard

Revisa cada punto y anota cualquier error o texto confuso.

**📋 Resumen**
- [ ] Se ven 4 indicadores: 4,238 pacientes · 15.2 % · 79.8 % · 0.713.
- [ ] Se ve el ciclo de datos (7 pasos) y la gráfica de riesgo por edad (4.1 % → 27.7 %).

**🔎 Exploración de datos**
- [ ] Cambiar *Sexo* a "Hombres": los indicadores y las gráficas cambian.
- [ ] Mover el rango de *Edad* a 60–70: el % con CHD sube (≈ 27–28 %).
- [ ] El selector de *Variable numérica* cambia la gráfica de cajas.
- [ ] *Ver datos de la selección* muestra la tabla.

**📈 Desempeño del modelo**
- [ ] Con el umbral en 0.12 la matriz muestra **365 / 354 / 26 / 103**.
- [ ] Al mover el umbral a 0.50, la sensibilidad baja a ≈ 8 % y el punto naranja de la curva ROC se mueve.
- [ ] Se ven la gráfica de calibración, los odds ratios y la tabla base vs. final.

**🧮 Calculadora de riesgo**
- [ ] Botón *Mujer joven saludable* → riesgo **bajo** (verde).
- [ ] Botón *Hombre de 55 años, fumador* → riesgo **elevado** (≈ 26 %).
- [ ] Botón *Adulto mayor con varios factores* → riesgo **muy elevado** (≈ 74 %), y la simulación baja a ≈ 31 %.
- [ ] Al apagar *Fuma actualmente*, el control de cigarrillos se desactiva y el riesgo baja.

---

## 4. Publicar el dashboard en internet (Streamlit Community Cloud)

**¿Por qué publicarlo?** Así la entrega incluye un **enlace** que el docente puede abrir desde cualquier navegador, sin instalar nada, y la presentación no depende de una sola laptop. Es **gratuito**.

**Requisitos:** una cuenta de **GitHub** (https://github.com/signup). El repositorio debe ser **público** para el plan gratuito.

### 4.1 Subir el proyecto a GitHub (sin usar comandos)

- [ ] Inicia sesión en GitHub → botón **New** (nuevo repositorio).
- [ ] Nombre sugerido: `riesgo-coronario-framingham` · Visibilidad: **Public** · **sin** README (ya tenemos archivos) → **Create repository**.
- [ ] En la página del repositorio vacío, da clic en **"uploading an existing file"**.
- [ ] Arrastra **estos** archivos y carpetas desde la carpeta del proyecto:
  - `Proyecto_Regresion_Logistica_CHD.ipynb`
  - `Proyecto_Regresion_Logistica_CHD.html` (si ya lo exportaste)
  - `requirements.txt`, `.gitignore`, `GUIA_REPLICACION.md`
  - Carpetas `dashboard/`, `data/`, `models/`, `figures/`, `.streamlit/`
- [ ] **No** subas: `.venv/`, `.claude/`, `.ipynb_checkpoints/`, `logistic-regression-heart-diseases-prediction.ipynb` ni el `framingham.csv` suelto de la raíz.
- [ ] Mensaje del commit: `Proyecto final: notebook, modelo y dashboard` → **Commit changes**.

> **Archivos ocultos:** en Windows, las carpetas que empiezan con punto (`.streamlit`) se ven normalmente; en Mac presiona **Cmd + Shift + .** en el Finder para mostrarlas. Verifica en GitHub que exista `.streamlit/config.toml`; si no aparece, créalo con **Add file → Create new file**, escribe `.streamlit/config.toml` como nombre y pega su contenido.

- [ ] Comprueba en GitHub que existan `models/modelo_regresion_logistica.pkl` y `data/processed/framingham_limpio.csv`; sin ellos el dashboard no funciona.

*(Alternativa con Git, si lo conoces: `git init`, `git add .`, `git commit -m "Proyecto final"`, y después `git remote add origin <url>` y `git push -u origin main`. El `.gitignore` excluye automáticamente lo que no debe subirse.)*

### 4.2 Crear la aplicación en Streamlit Community Cloud

- [ ] Entra a https://share.streamlit.io e inicia sesión con **GitHub** (autoriza el acceso cuando lo pida).
- [ ] Clic en **Create app** → **Deploy a public app from GitHub** (o *"Yup, I have an app"*).
- [ ] Llena el formulario:

| Campo | Valor |
|---|---|
| Repository | `tu-usuario/riesgo-coronario-framingham` |
| Branch | `main` |
| Main file path | `dashboard/app.py` |
| App URL (opcional) | por ejemplo `riesgo-coronario-framingham` |

- [ ] Abre **Advanced settings** → **Python version**: elige **3.12** (o la más reciente que ofrezca). Guarda.
- [ ] Clic en **Deploy**. La primera vez tarda **3–5 minutos** instalando librerías (puedes ver el avance en el panel de *logs* de la derecha).
- [ ] Al terminar se abre el dashboard con una dirección del tipo `https://riesgo-coronario-framingham.streamlit.app`.
- [ ] Repite la **lista de pruebas de la sección 3.1** sobre la versión publicada.
- [ ] Comparte el enlace con el equipo.

> Streamlit Cloud usa automáticamente `dashboard/requirements.txt` (está junto a `app.py`), que contiene solo lo necesario para el dashboard.

### 4.3 Antes de la presentación y de la entrega

- [ ] Las apps gratuitas **se "duermen"** tras varios días sin visitas. **El día de la presentación, ábrela 10 minutos antes**; si muestra *"This app has gone to sleep"*, da clic en **"Yes, get this app back up!"** y espera 1–2 minutos.
- [ ] Ten como **respaldo** el dashboard corriendo localmente (sección 3) en la laptop de la presentación.
- [ ] Si se modifica el notebook o el dashboard, sube los archivos actualizados a GitHub: la app se actualiza sola en 1–2 minutos.

---

## 5. Problemas comunes

| Problema | Solución |
|---|---|
| `python` no se reconoce como comando | Reinstala Python marcando "Add Python to PATH", o usa `py` (Windows) / `python3` (Mac). |
| `ModuleNotFoundError: No module named ...` | El entorno virtual no está activado, o falta instalar: repite 1.3 y 1.4. |
| El dashboard dice *"No se encontraron los archivos del modelo"* | Ejecuta el notebook completo (sección 2) para generar `models/` y `data/processed/`; en la nube, verifica que esas carpetas estén en GitHub. |
| Error o advertencia al cargar el `.pkl` (`InconsistentVersionWarning`, `AttributeError`) | La versión de scikit-learn no coincide. Instala exactamente las de `requirements.txt`, o vuelve a ejecutar el notebook para regenerar el modelo con tu versión. |
| El dashboard se ve sin colores/tema | Lo ejecutaste desde dentro de `dashboard/`; ejecútalo desde la carpeta raíz. |
| En Streamlit Cloud falla la instalación | Revisa los *logs*: si una librería no tiene versión para ese Python, cambia la versión de Python en *Settings → General* (3.12 suele funcionar) y da clic en *Reboot app*. |
| Los números del notebook no coinciden con la tabla 2.3 | Verifica las versiones de las librerías y que ejecutaste **todas** las celdas en orden desde un kernel reiniciado. |

---

## 6. Qué se entrega

- [ ] **Proyecto técnico:**
  - `Proyecto_Regresion_Logistica_CHD.ipynb` (ejecutado, con todas las salidas visibles);
  - su versión `.html`;
  - las carpetas `data/`, `models/`, `figures/` y `requirements.txt`.
- [ ] **Dashboard:**
  - el enlace público de Streamlit Cloud;
  - el código en `dashboard/app.py`.
- [ ] **Repositorio de GitHub:** enlace con todo lo anterior.
- [ ] **Presentación ejecutiva.**
- [ ] Opcional: un `.zip` de la carpeta del proyecto (sin `.venv/`), como respaldo por si el docente no puede acceder a los enlaces.
