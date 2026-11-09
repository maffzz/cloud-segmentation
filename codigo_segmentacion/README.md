# Segmentación multi-etapa de una nube de puntos industrial

Este proyecto implementa una solución completa para la segmentación semántica multi-etapa de una nube de puntos 3D densa (`FARO_Scan_Full_Sub1_down3_with_normals.ply`, 1.647.021 puntos) correspondiente a una planta industrial.

Aprovecha de forma desacoplada y secuencial los atributos de **posición espacial 3D (XYZ)**, **colorimetría (RGB / HSV)** y **orientación de normales superficiales ($n_x, n_y, n_z$)**.

---

## 1. Requisitos e Instalación

Entorno recomendado: Python 3.9 a 3.12.

Instalar dependencias necesarias:
```bash
python3 -m pip install -r requirements.txt
```

Dependencias principales:
* `open3d`: Carga, inspección y exportación de nubes de puntos PLY.
* `numpy` y `scipy`: Álgebra lineal, KD-Trees espaciales (`cKDTree`), operaciones morfológicas y análisis de picos.
* `scikit-learn`: Algoritmos de clustering espacial basados en densidad (`DBSCAN`).
* `matplotlib`: Renderizado y generación de vistas ortogonales, oblicuas y acercamientos.

---

## 2. Instrucciones de Ejecución

Desde este directorio (`codigo_segmentacion/`):

```bash
# 1. Crear carpeta de salida
mkdir -p out

# 2. Inspeccionar estadísticas geométricas y radiométricas de la nube cruda
python3 src/inspect_cloud.py ../FARO_Scan_Full_Sub1_down3_with_normals.ply

# 3. Ejecutar pipeline completo de segmentación y exportar PLY resultante
python3 src/segment_cloud.py ../FARO_Scan_Full_Sub1_down3_with_normals.ply ../nube_segmentada_completa.ply --stats out/stats.json

# 4. Renderizar vistas generales (oblicua 1, oblicua 2, alzado, perfil y planta)
python3 src/render_views.py ../FARO_Scan_Full_Sub1_down3_with_normals.ply out/original_rgb
python3 src/render_views.py ../nube_segmentada_completa.ply out/final

# 5. Generar comparativa lado a lado y figuras de detalle/acercamiento
python3 src/render_enhanced.py
```

El tiempo de cómputo total en un equipo estándar es de aproximadamente **60 segundos** para el millón y medio de puntos.

---

## 3. Visualización en Software 3D (CloudCompare / MeshLab)

El archivo exportado `nube_segmentada_completa.ply` preserva estrictamente los 1.647.021 puntos originales con sus coordenadas XYZ y normales, incorporando los colores RGB asignados a cada clase y una propiedad entera adicional llamada `label` (0 a 12).

### En CloudCompare:
1. Abrir `nube_segmentada_completa.ply` (`File -> Open`).
2. En el diálogo emergente de carga PLY, asegurarse de que las propiedades `x, y, z`, `red, green, blue`, `nx, ny, nz` y el escalar `label` queden seleccionados.
3. Para ver las clases por color: En el panel de **Properties -> Colors**, seleccionar `RGB`.
4. Para inspeccionar o filtrar por clase numérica:
   * En **Properties -> Scalar Fields**, seleccionar `label`.
   * En el menú superior de paletas de color, elegir una rampa discreta (p. ej. `Color ramp: Blue > Green > Yellow > Red`).
   * Usar la herramienta **Filter points by value** para aislar clases específicas (p. ej., ver exclusivamente las tuberías con `label = 8` o columnas con `label = 5`).
5. Para máxima calidad visual: Activar el sombreado **Display -> Eye-Dome Lighting (EDL)**.

---

## 4. Estructura del Pipeline Metodológico

1. **E1. Planos horizontales:** Análisis de histograma de elevación $Z$ sobre puntos con $|n_z| > 0.9$ y segmentación espacial DBSCAN $\to$ *Piso, Techo y Plataformas*.
2. **E2. Planos verticales:** RANSAC iterativo con restricciones de normal horizontal ($|n_z| < 0.2$) $\to$ *Paredes / Paneles*.
3. **E3. Tensor local de normales (Multi-escala):** Matriz de segundo momento $M = \frac{1}{k}\sum n_i n_i^T$. Descomposición en autovalores a escala fina ($k=50$) y gruesa (~25 cm) para detectar linealidad/extrusión y estimar el vector director del eje.
4. **E4. Elementos estructurales y mecánicos extruidos:** DBSCAN por tramos alineados según la orientación principal del eje (Vertical, X, Y, Diagonal). Clasificación fina mediante análisis de armónicos angulares de la sección transversal ($h_2, h_4$) y espacio de color HSV $\to$ *Tuberías cilíndricas, Columnas verticales, Vigas ortogonales y Arriostres diagonales*.
5. **E4b. Miembros lineales largos:** Detección sobre rejilla de vóxeles 3D con clausura morfológica unidireccional y crecimiento de regiones (*Region Growing* con KD-Tree).
6. **E5. Cuerpos compactos restantes:** DBSCAN en espacio mixto métrico-radiométrico (XYZ + peso de color RGB) $\to$ *Tanques/Equipos industriales y Accesorios menores*.
7. **E6. Regularización espacial:** Suavizado de bordes y reducción de ruido mediante votación mayoritaria ponderada por $k$-vecinos más cercanos ($k$-NN).

---

## 5. Tabla de Códigos de Etiqueta (`label`)

| `label` | Nombre de Clase | Color RGB de Representación |
| :---: | :--- | :---: |
| 0 | Sin clasificar | (150, 150, 150) |
| 1 | Piso | (30, 60, 230) |
| 2 | Techo | (0, 190, 210) |
| 3 | Plataforma / entrepiso | (140, 70, 210) |
| 4 | Pared / panel | (70, 120, 175) |
| 5 | Columna | (0, 170, 0) |
| 6 | Viga | (255, 140, 0) |
| 7 | Arriostre (diagonal) | (240, 230, 0) |
| 8 | Tubería | (225, 0, 0) |
| 9 | Equipo / tanque | (255, 0, 255) |
| 10 | Persona *(regla heurística condicional)* | (255, 255, 255) |
| 11 | Vehículo *(regla heurística condicional)* | (140, 80, 20) |
| 12 | Accesorio / equipo menor | (255, 150, 170) |
