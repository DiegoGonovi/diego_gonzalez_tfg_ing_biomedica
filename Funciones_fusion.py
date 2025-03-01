import os
from ultralytics import YOLO
import glob
import nibabel as nib
import plotly.graph_objects as go
from PIL import Image
import numpy as np
import cv2
import os
import ipynb
import matplotlib.pyplot as plt


### Etapa de fusión
# %%
def gt_voi_in_txt_2(volumes, output, patience):
    """
    Guarda los volúmenes en un archivo de texto, cada volumen en una línea.
    Si la lista está vacía, genera un archivo vacío sin contenido.

    Parámetros:
      - volumes: Lista de arrays de forma (8,3), cada uno representando un volumen.
      - output: Carpeta donde se guardará el archivo.
      - patience: Identificador del paciente.
    """

    # Definir la ruta de salida
    output_path = os.path.join(output, f"{patience}_gt.txt")

    with open(output_path, 'w') as file:
        if len(volumes) > 0:  # Solo escribir si hay volúmenes
            for vol in volumes:
                contenido = ' '.join(map(str, vol.flatten()))  # Convertir todo el volumen en una única línea
                file.write(contenido + "\n")  # Escribirlo en el archivo en una sola línea por volumen

    print(f"Archivo guardado en: {output_path} {'(vacío)' if len(volumes) == 0 else ''}")


# %%
def txt_array_gt(output_path, patience, filas, columnas):
    """
    Lee un archivo de texto que contiene datos de volúmenes y los convierte en un array numpy con la forma especificada.

    Parámetros:
      - output_path: Carpeta donde se encuentra el archivo de texto.
      - patience: Identificador del paciente.
      - filas: Número de filas de la matriz resultante.
      - columnas: Número de columnas de la matriz resultante.

    Return:
      - Un array numpy de forma (filas, columnas) con los datos leídos del archivo.
    """
    # Define la ruta de entrada usando la variable `pat`
    input_path = os.path.join(output_path, f"{patience}_gt.txt")

    # Lee el contenido del archivo y convierte a una lista de enteros
    with open(input_path, 'r') as file:
        contenido = file.read().split(' ')

    # Convierte la lista en un array numpy y le da la forma especificada
    array = np.array(list(map(int, contenido))).reshape(filas, columnas)

    return array


# %%
def generate_voi(axial, coronal, sagital):
    """
    Genera un VOI a partir de la intersección de los planos axial, coronal y sagital.
    Devuelve un array de 8 coordenadas (los vértices del cubo resultante de la intersección).

    Parámetros:
      - axial: Array de coordenadas del plano axial, de forma (N, 3).
      - coronal: Array de coordenadas del plano coronal, de forma (N, 3).
      - sagital: Array de coordenadas del plano sagital, de forma (N, 3).

    Return:
      - Array numpy de forma (8,3) con las coordenadas de los 8 vértices del VOI.

    """
    # Obtener las coordenadas de intersección entre los tres planos
    x_min = axial[:, 0].min()
    x_max = axial[:, 0].max()

    y_min = coronal[:, 1].min()
    y_max = coronal[:, 1].max()

    z_min = sagital[:, 2].min()
    z_max = sagital[:, 2].max()

    # Crear las 8 coordenadas de la intersección (cubo)
    return np.array([
        [x_min, y_min, z_min],
        [x_min, y_min, z_max],
        [x_min, y_max, z_min],
        [x_min, y_max, z_max],
        [x_max, y_min, z_min],
        [x_max, y_min, z_max],
        [x_max, y_max, z_min],
        [x_max, y_max, z_max]
    ])


# %%
def find_vois(corners_sagital, corners_coronal, corners_axial):
    """
    Encuentra los Volúmenes de Interés (VOIs) a partir de la intersección de las cajas delimitadoras
    en los planos sagital, coronal y axial.

    Parámetros:
      - corners_sagital: Lista de arrays (8,3) con los vértices de las cajas delimitadoras en el plano sagital.
      - corners_coronal: Lista de arrays (8,3) con los vértices de las cajas delimitadoras en el plano coronal.
      - corners_axial: Lista de arrays (8,3) con los vértices de las cajas delimitadoras en el plano axial.

    Return:
      - Una lista de arrays (8,3), donde cada array representa un VOI único detectado.
    """

    bb_corners_axial = [np.array(bbox) for bbox in corners_axial]
    bb_corners_coronal = [np.array(bbox) for bbox in corners_coronal]
    bb_corners_sagital = [np.array(bbox) for bbox in corners_sagital]

    vois = []

    for bb_axial in bb_corners_axial:
        in_coronal = False
        in_sagital = False
        for bb_coronal in bb_corners_coronal:

            # Verifica que esté dentro de los límites de la caja coronal
            for_x_axial = bb_axial[:, 0].min() <= bb_coronal[:, 0].min() <= bb_axial[:, 0].max()

            for_y_coronal = (bb_coronal[:, 1].min() <= bb_axial[:, 1].min() <= bb_coronal[:, 1].max() or
                             bb_axial[:, 1].min() <= bb_coronal[:, 1].max() <= bb_axial[:, 1].max())

            for_z_coronal = bb_coronal[:, 2].min() <= bb_axial[:, 2].min() <= bb_coronal[:, 2].max()

            if for_y_coronal and for_z_coronal and for_x_axial:  # Dentro del coronal
                in_coronal = True

                for bb_sagital in bb_corners_sagital:

                    for_x_sagital = (
                                bb_sagital[:, 0].min() <= bb_axial[:, 0].min() <= bb_sagital[:, 0].max() or
                                bb_axial[:,0].min() <= bb_sagital[:,0].max() <= bb_axial[:,0].max())

                    for_z_sagital = bb_sagital[:, 2].min() <= bb_axial[:, 2].min() <= bb_sagital[:, 2].max()

                    for_y_axial = bb_axial[:, 1].min() <= bb_sagital[:, 1].min() <= bb_axial[:, 1].max()

                    if for_x_sagital and for_z_sagital and for_y_axial:
                        in_sagital = True
                        voi = generate_voi(bb_axial, bb_coronal, bb_sagital)
                        if not any(np.array_equal(existing_voi, voi) for existing_voi in vois):
                            vois.append(voi)
    return vois


# %%
def axial_coronal_voi(corners_coronal, corners_axial):
    """
    Encuentra los Volúmenes de Interés (VOIs) a partir de la intersección de las cajas delimitadoras
    en los planos axial y coronal.

    Parámetros:
      - corners_coronal: Lista de arrays (8,3) con los vértices de las cajas delimitadoras en el plano coronal.
      - corners_axial: Lista de arrays (8,3) con los vértices de las cajas delimitadoras en el plano axial.

    Return:
      - Una lista de arrays (8,3), donde cada array representa un VOI único detectado.
    """

    bb_corners_axial = [np.array(bbox) for bbox in corners_axial]
    bb_corners_coronal = [np.array(bbox) for bbox in corners_coronal]
    vois = []

    for bb_axial in bb_corners_axial:
        for bb_coronal in bb_corners_coronal:
            # Verifica que esté dentro de los límites de la caja coronal
            for_x_axial = bb_axial[:, 0].min() <= bb_coronal[:, 0].min() <= bb_axial[:, 0].max()

            for_y_coronal = (bb_coronal[:, 1].min() <= bb_axial[:, 1].min() <= bb_coronal[:, 1].max() or
                             bb_coronal[:, 1].min() <= bb_axial[:, 1].max() <= bb_coronal[:, 1].max() or
                             bb_axial[:, 1].min() <= bb_coronal[:, 1].max() <= bb_axial[:, 1].max())

            for_z_coronal = bb_coronal[:, 2].min() <= bb_axial[:, 2].min() <= bb_coronal[:, 2].max()

            if for_y_coronal and for_z_coronal and for_x_axial:  # Dentro del coronal
                voi = generate_voi(bb_axial, bb_coronal, bb_coronal)
                if not any(np.array_equal(existing_voi, voi) for existing_voi in vois):
                    vois.append(voi)

    return vois


# %%
def axial_sagital_voi(corners_sagital, corners_axial):
    """
    Encuentra los Volúmenes de Interés (VOIs) a partir de la intersección de las cajas delimitadoras
    en los planos axial y sagital.

    Parámetros:
      - corners_sagital: Lista de arrays (8,3) con los vértices de las cajas delimitadoras en el plano sagital.
      - corners_axial: Lista de arrays (8,3) con los vértices de las cajas delimitadoras en el plano axial.

    Return:
      - Una lista de arrays (8,3), donde cada array representa un VOI único detectado.
    """

    bb_corners_axial = [np.array(bbox) for bbox in corners_axial]
    bb_corners_sagital = [np.array(bbox) for bbox in corners_sagital]
    vois = []

    for bb_axial in bb_corners_axial:
        for bb_sagital in bb_corners_sagital:

            for_x_sagital = (bb_sagital[:, 0].min() <= bb_axial[:, 0].min() <= bb_sagital[:, 0].max() or
                             bb_axial[:,0].min() <= bb_sagital[:,0].max() <= bb_axial[:,0].max())

            for_z_sagital = bb_sagital[:, 2].min() <= bb_axial[:, 2].min() <= bb_sagital[:, 2].max()

            for_y_axial = bb_axial[:, 1].min() <= bb_sagital[:, 1].min() <= bb_axial[:, 1].max()

            if for_x_sagital and for_z_sagital and for_y_axial:
                in_sagital = True
                voi = generate_voi(bb_axial, bb_axial, bb_sagital)
                if not any(np.array_equal(existing_voi, voi) for existing_voi in vois):
                    vois.append(voi)
    return vois


# %%
def coronal_sagital(corners_coronal, corners_sagital):
    """
    Encuentra los Volúmenes de Interés (VOIs) a partir de la intersección de las cajas delimitadoras
    en los planos coronal y sagital.

    Parámetros:
      - corners_coronal: Lista de arrays (8,3) con los vértices de las cajas delimitadoras en el plano coronal.
      - corners_sagital: Lista de arrays (8,3) con los vértices de las cajas delimitadoras en el plano sagital.

    Return:
      - Una lista de arrays (8,3), donde cada array representa un VOI único detectado.
    """

    bb_corners_coronal = [np.array(bbox) for bbox in corners_coronal]
    bb_corners_sagital = [np.array(bbox) for bbox in corners_sagital]
    vois = []

    for bb_sagital in bb_corners_sagital:
        for bb_coronal in bb_corners_coronal:

            for_x_coronal = (bb_sagital[:, 0].min() <= bb_coronal[:, 0].min() <= bb_sagital[:, 0].max() or bb_coronal[:,
                                                                                                           0].min() <= bb_sagital[
                                                                                                                       :,
                                                                                                                       0].max() <= bb_coronal[
                                                                                                                                   :,
                                                                                                                                   0].max())

            for_y_coronal = bb_coronal[:, 1].min() <= bb_sagital[:, 1].min() <= bb_coronal[:, 1].max()

            for_z_sagital = bb_sagital[:, 2].min() <= bb_coronal[:, 2].min() <= bb_sagital[:, 2].max()

            if for_x_coronal and for_z_sagital and for_y_coronal:
                voi = generate_voi(bb_sagital, bb_coronal, bb_sagital)
                if not any(np.array_equal(existing_voi, voi) for existing_voi in vois):
                    vois.append(voi)
    return vois


# %%
def full_one_voi(vois):
    """
    Genera un único VOI que engloba todos los VOIs dados en la lista,
    calculando los límites mínimos y máximos en cada dimensión.

    Parámetros:
      - vois: Lista de arrays (8,3), donde cada array representa un VOI.

    Retorna:
      - Un array numpy (8,3) representando el VOI que contiene todos los VOIs de entrada.
    """

    all_points = np.vstack(vois)  # Combina todos los puntos en un solo array para facilitar el cálculo

    # Calcula los límites mínimo y máximo en cada dimensión (x, y, z)
    min_x, min_y, min_z = all_points.min(axis=0)
    max_x, max_y, max_z = all_points.max(axis=0)

    # Generar el VOI que engloba a todos los VOIs en `vois`
    voi_enclosing_all = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z]
    ])

    return voi_enclosing_all


# %%
def coord_txt_dic(folder_root, corte):
    """
    Genera un diccionario de coordenadas basado en archivos de texto en una carpeta.

    Parámetros:
    - folder_root: Ruta de la carpeta que contiene los archivos.
    - corte: Tipo de corte ('axial' o 'coronal') que determina la organización de las coordenadas.

    Return:
    - Un diccionario con nombres de pacientes como claves y listas de coordenadas como valores.
    """
    labels = {}
    archivos_txt = os.listdir(folder_root)
    for i in archivos_txt:
        full_file = os.path.join(folder_root, i)
        patient_path = full_file.split("\\")[-1].split('_')
        patient_name = patient_path[0]

        value_z = full_file.split('_')[-1].split('.')
        value_z = int(value_z[0])

        # Inicializa una lista para almacenar las coordenadas de este paciente
        coordenadas = []

        with (open(full_file, "r") as file):
            for line in file:
                x1, y1, x2, y2 = [int(float(item)) for item in line.split()]
                # Asigna coordenadas según el tipo de corte

                if corte == "axial":
                    corners = [
                        [x1, y1, value_z],
                        [x2, y1, value_z],
                        [x2, y2, value_z],
                        [x1, y2, value_z]
                    ]
                elif corte == "coronal":
                    corners = [
                        [value_z, y1, x1],
                        [value_z, y1, x2],
                        [value_z, y2, x2],
                        [value_z, y2, x1]
                    ]

                elif corte == "sagital":
                    corners = [
                        [y1, value_z, x1],
                        [y1, value_z, x2],
                        [y2, value_z, x2],
                        [y2, value_z, x1]
                    ]

                coordenadas.append(corners)

        # Asigna las coordenadas al diccionario usando el nombre del paciente como clave
        if patient_name not in labels:
            labels[patient_name] = []  # Inicializa la lista si no existe

        labels[patient_name].extend(coordenadas)  # Agrega las coordenadas a la lista del paciente

    return labels


# %%
def plot_superficies_planas_multi(paciente, coordenadas_axial, coordenadas_coronal, coordenadas_sagital):
    """
    Genera una visualización 3D interactiva con superficies planas correspondientes
    a los planos axial, coronal y sagital usando Plotly.

    Parámetros:
      - paciente: Identificador del paciente, usado en el título de la figura.
      - coordenadas_axial: Lista de arrays (8,3) con los vértices de las superficies en el plano axial.
      - coordenadas_coronal: Lista de arrays (8,3) con los vértices de las superficies en el plano coronal.
      - coordenadas_sagital: Lista de arrays (8,3) con los vértices de las superficies en el plano sagital.

    """

    # Crear la figura de Plotly
    fig = go.Figure()

    # Añadir superficies axiales en color rojo
    for vertices in coordenadas_axial:
        x = [v[0] for v in vertices]
        y = [v[1] for v in vertices]
        z = [v[2] for v in vertices]

        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            color='red',
            opacity=0.5,
            i=[0, 0, 0, 1, 1, 1],
            j=[1, 2, 3, 2, 3, 0],
            k=[2, 3, 0, 3, 0, 1],
            showlegend=False
        ))

    # Añadir superficies coronales en color verde
    for vertices in coordenadas_coronal:
        x = [v[0] for v in vertices]
        y = [v[1] for v in vertices]
        z = [v[2] for v in vertices]

        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            color='green',
            opacity=0.5,
            i=[0, 0, 0, 1, 1, 1],
            j=[1, 2, 3, 2, 3, 0],
            k=[2, 3, 0, 3, 0, 1],
            showlegend=False
        ))
    # Añadir superficies sagitales en azul
    for vertices in coordenadas_sagital:
        x = [v[0] for v in vertices]
        y = [v[1] for v in vertices]
        z = [v[2] for v in vertices]

        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            color='blue',
            opacity=0.5,
            i=[0, 0, 0, 1, 1, 1],
            j=[1, 2, 3, 2, 3, 0],
            k=[2, 3, 0, 3, 0, 1],
            showlegend=False
        ))

    # Ajustar el diseño de la gráfica
    fig.update_layout(scene=dict(
        xaxis=dict(title='X', backgroundcolor='rgba(0,0,0,0)', showgrid=True),
        yaxis=dict(title='Y', backgroundcolor='rgba(0,0,0,0)', showgrid=True),
        zaxis=dict(title='Z', backgroundcolor='rgba(0,0,0,0)', showgrid=True)
    ),
        title=f'Superficies planas en el espacio 3D - Paciente {paciente}',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)')

    # Mostrar la figura en el navegador
    fig.show()


# %%
def plot_voi_3_slices(paciente, coordenadas_axial, coordenadas_coronal, coordenadas_sagital, vois):
    """
    Genera una visualización 3D de los planos axiales, coronales y sagitales junto con los Volúmenes de Interés (VOIs).

    Parámetros:
      - paciente: Identificador del paciente para el título de la gráfica.
      - coordenadas_axial: Lista de arrays (8,3) con los vértices de las superficies axiales.
      - coordenadas_coronal: Lista de arrays (8,3) con los vértices de las superficies coronales.
      - coordenadas_sagital: Lista de arrays (8,3) con los vértices de las superficies sagitales.
      - vois: Lista de arrays (8,3), donde cada array representa un VOI.

    """

    faces = [
        [0, 1, 3, 2],  # Cara del frente
        [4, 5, 7, 6],  # Cara de atrás
        [0, 1, 5, 4],  # Cara de abajo
        [2, 3, 7, 6],  # Cara de arriba
        [0, 2, 6, 4],  # Cara izquierda
        [1, 3, 7, 5]  # Cara derecha
    ]

    # Crear la lista de trazas para todos los cubos
    traces = []

    # Iterar sobre cada cubo en vois
    for cube in vois:
        for face in faces:
            # Conectamos los vértices de la cara del cubo actual
            x_vals = [cube[face[0]][0], cube[face[1]][0], cube[face[2]][0], cube[face[3]][0], cube[face[0]][0]]
            y_vals = [cube[face[0]][1], cube[face[1]][1], cube[face[2]][1], cube[face[3]][1], cube[face[0]][1]]
            z_vals = [cube[face[0]][2], cube[face[1]][2], cube[face[2]][2], cube[face[3]][2], cube[face[0]][2]]

            traces.append(go.Scatter3d(
                x=x_vals, y=y_vals, z=z_vals,
                mode='lines',
                line=dict(color='rgb(255, 102, 0)', width=4)
            ))

    # Crear la figura de Plotly
    fig = go.Figure()

    # Añadir superficies axiales en color rojo
    for vertices in coordenadas_axial:
        x = [v[0] for v in vertices]
        y = [v[1] for v in vertices]
        z = [v[2] for v in vertices]

        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            color='red',
            opacity=0.5,
            i=[0, 1, 2],
            j=[1, 2, 3],
            k=[2, 3, 0],
            showlegend=False
        ))

    # Añadir superficies coronales en color verde
    for vertices in coordenadas_coronal:
        x = [v[0] for v in vertices]
        y = [v[1] for v in vertices]
        z = [v[2] for v in vertices]

        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            color='green',
            opacity=0.5,
            i=[0, 1, 2],
            j=[1, 2, 3],
            k=[2, 3, 0],
            showlegend=False
        ))

    # Añadir superficies sagitales en azul
    for vertices in coordenadas_sagital:
        x = [v[0] for v in vertices]
        y = [v[1] for v in vertices]
        z = [v[2] for v in vertices]

        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            color='blue',
            opacity=0.5,
            i=[0, 1, 2],
            j=[1, 2, 3],
            k=[2, 3, 0],
            showlegend=False
        ))

    # Añadir las trazas de los cubos (VOIs)
    fig.add_traces(traces)

    # Ajustar el diseño de la gráfica
    fig.update_layout(scene=dict(
        xaxis=dict(title='X', backgroundcolor='rgba(0,0,0,0)', showgrid=True),
        yaxis=dict(title='Y', backgroundcolor='rgba(0,0,0,0)', showgrid=True),
        zaxis=dict(title='Z', backgroundcolor='rgba(0,0,0,0)', showgrid=True)
    ),
        title=f'Superficies planas y su VOI en el espacio 3D - Paciente {paciente}',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)', showlegend=False)

    # Mostrar la figura en el navegador
    fig.show()


# %%
def plot_2_vois_3_slices(paciente, coordenadas_axial, coordenadas_coronal, coordenadas_sagital, vois, gt_voi):
    """
    Genera una visualización 3D de los planos axiales, coronales y sagitales, junto con los Volúmenes de Interés (VOIs)
    detectados y el VOI de referencia (GT VOI).

    Parámetros:
      - paciente: Identificador del paciente para el título de la gráfica.
      - coordenadas_axial: Lista de arrays (8,3) con los vértices de las superficies axiales.
      - coordenadas_coronal: Lista de arrays (8,3) con los vértices de las superficies coronales.
      - coordenadas_sagital: Lista de arrays (8,3) con los vértices de las superficies sagitales.
      - vois: Lista de arrays (8,3), donde cada array representa un VOI detectado.
      - gt_voi: Array (8,3) con los vértices del VOI de referencia (Ground Truth).

    """
    faces = [
        [0, 1, 3, 2],  # Cara del frente
        [4, 5, 7, 6],  # Cara de atrás
        [0, 1, 5, 4],  # Cara de abajo
        [2, 3, 7, 6],  # Cara de arriba
        [0, 2, 6, 4],  # Cara izquierda
        [1, 3, 7, 5]  # Cara derecha
    ]

    # Crear la lista de trazas para todos los cubos
    traces = []

    # Iterar sobre cada cubo en vois
    for cube in vois:
        for face in faces:
            x_vals = [cube[face[0]][0], cube[face[1]][0], cube[face[2]][0], cube[face[3]][0], cube[face[0]][0]]
            y_vals = [cube[face[0]][1], cube[face[1]][1], cube[face[2]][1], cube[face[3]][1], cube[face[0]][1]]
            z_vals = [cube[face[0]][2], cube[face[1]][2], cube[face[2]][2], cube[face[3]][2], cube[face[0]][2]]

            traces.append(go.Scatter3d(
                x=x_vals, y=y_vals, z=z_vals,
                mode='lines',
                line=dict(color='rgb(255, 255, 0)', width=6)
            ))

    # Añadir las caras del gt_voi en rosa chillón
    for face in faces:
        x_vals = [gt_voi[face[0]][0], gt_voi[face[1]][0], gt_voi[face[2]][0], gt_voi[face[3]][0], gt_voi[face[0]][0]]
        y_vals = [gt_voi[face[0]][1], gt_voi[face[1]][1], gt_voi[face[2]][1], gt_voi[face[3]][1], gt_voi[face[0]][1]]
        z_vals = [gt_voi[face[0]][2], gt_voi[face[1]][2], gt_voi[face[2]][2], gt_voi[face[3]][2], gt_voi[face[0]][2]]

        traces.append(go.Scatter3d(
            x=x_vals, y=y_vals, z=z_vals,
            mode='lines',
            line=dict(color='rgb(255, 102, 0)', width=6)  # Rosa chillón
        ))

    # Crear la figura de Plotly
    fig = go.Figure()

    # Añadir superficies axiales en color rojo
    for vertices in coordenadas_axial:
        x = [v[0] for v in vertices]
        y = [v[1] for v in vertices]
        z = [v[2] for v in vertices]

        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            color='red',
            opacity=0.5,
            i=[0, 1, 2],
            j=[1, 2, 3],
            k=[2, 3, 0],
            showlegend=False
        ))

    # Añadir superficies coronales en color verde
    for vertices in coordenadas_coronal:
        x = [v[0] for v in vertices]
        y = [v[1] for v in vertices]
        z = [v[2] for v in vertices]

        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            color='green',
            opacity=0.5,
            i=[0, 1, 2],
            j=[1, 2, 3],
            k=[2, 3, 0],
            showlegend=False
        ))

    # Añadir superficies sagitales en azul
    for vertices in coordenadas_sagital:
        x = [v[0] for v in vertices]
        y = [v[1] for v in vertices]
        z = [v[2] for v in vertices]

        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            color='blue',
            opacity=0.5,
            i=[0, 1, 2],
            j=[1, 2, 3],
            k=[2, 3, 0],
            showlegend=False
        ))

    # Añadir las trazas de los cubos (VOIs y GT VOI)
    fig.add_traces(traces)

    # Ajustar el diseño de la gráfica
    fig.update_layout(scene=dict(
        xaxis=dict(title='X', backgroundcolor='rgba(0,0,0,0)', showgrid=True),
        yaxis=dict(title='Y', backgroundcolor='rgba(0,0,0,0)', showgrid=True),
        zaxis=dict(title='Z', backgroundcolor='rgba(0,0,0,0)', showgrid=True)
    ),
        title=f'Superficies planas y su VOI en el espacio 3D - Paciente {paciente}',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)', showlegend=False)

    # Mostrar la figura en el navegador
    fig.show()


# %%
def yolo_detections(lst_images, model_yolo, corte):
    """
    Realiza detecciones de objetos en un conjunto de imágenes usando un modelo YOLO y
    extrae las coordenadas de las cajas delimitadoras, ajustándolas al tipo de corte (axial, coronal o sagital).

    Parámetros:
      - lst_images: Ruta a la carpeta que contiene las imágenes a analizar.
      - model_yolo: Modelo YOLO preentrenado para realizar las detecciones.
      - corte: Tipo de corte en el que se encuentran las imágenes ('axial', 'coronal' o 'sagital').

    Return:
      - Lista de coordenadas de las cajas detectadas en formato de esquinas 3D.
    """
    coordenadas = []

    for file in os.listdir(lst_images):
        results = model_yolo(source=os.path.join(lst_images, file), conf=0.1, save=False, device=0)
        patient_name = file.split('.')[0]  # Nombre del paciente sin la extensión

        value_z = int(patient_name.split('_')[-1])  # Extraer la coordenada Z desde el nombre del archivo

        for result in results:
            boxes = result.boxes.xyxy.tolist()

            for box in boxes:
                x1, y1, x2, y2 = map(int, map(float, box))  # Convertir coordenadas a enteros

                # Generar esquinas según el tipo de corte
                if corte == "axial":
                    corners = [[x1, y1, value_z], [x2, y1, value_z], [x2, y2, value_z], [x1, y2, value_z]]
                elif corte == "coronal":
                    corners = [[value_z, y1, x1], [value_z, y1, x2], [value_z, y2, x2], [value_z, y2, x1]]
                elif corte == "sagital":
                    corners = [[y1, value_z, x1], [y1, value_z, x2], [y2, value_z, x2], [y2, value_z, x1]]

                coordenadas.append(corners)

    return coordenadas

#%%
def yolo_detections_full(lst_images, model_yolo, corte):
    """
    Realiza detecciones de objetos en un conjunto de imágenes usando un modelo YOLO y
    extrae las coordenadas de las cajas delimitadoras, ajustándolas al tipo de corte (axial, coronal o sagital).

    Parámetros:
      - lst_images: Lista de tuplas (coordenada Z, ruta de la imagen), donde:
          # file[0]: Coordenada Z asociada a la imagen.
          # file[1]: Ruta de la imagen en la que se realizará la detección.
      - model_yolo: Modelo YOLO preentrenado para realizar las detecciones.
      - corte: Tipo de corte en el que se encuentran las imágenes ('axial', 'coronal' o 'sagital').

    Return:
      - Lista de coordenadas de las cajas detectadas en formato de esquinas 3D.
    """
    coordenadas = []
    for file in lst_images:
        results = model_yolo(source=file[1], conf=0.1, save=False, device=0)
        value_z = file[0]  # Extraer la coordenada Z desde el nombre del archivo

        for result in results:
            boxes = result.boxes.xyxy.tolist()

            for box in boxes:
                x1, y1, x2, y2 = map(int, map(float, box))  # Convertir coordenadas a enteros

                # Generar esquinas según el tipo de corte
                if corte == "axial":
                    corners = [[x1, y1, value_z], [x2, y1, value_z], [x2, y2, value_z], [x1, y2, value_z]]
                elif corte == "coronal":
                    corners = [[value_z, y1, x1], [value_z, y1, x2], [value_z, y2, x2], [value_z, y2, x1]]
                elif corte == "sagital":
                    corners = [[y1, value_z, x1], [y1, value_z, x2], [y2, value_z, x2], [y2, value_z, x1]]

                coordenadas.append(corners)

    return coordenadas

# %%
def merge_vol(vois, margin):
    """
    Fusiona volúmenes de interés (VOIs) cercanos, agrupándolos si están dentro de un margen determinado.

    Parámetros:
      - vois: Lista de arrays (8,3), donde cada array representa un VOI.
      - margin: Margen de tolerancia para agrupar volúmenes cercanos.

    Return:
      - Lista de VOIs fusionados, cada uno representado como un array (8,3).
    """

    vois = [np.array(voi) for voi in vois]

    boundaries = [(voi.min(axis=0), voi.max(axis=0)) for voi in vois]
    merged = []

    while boundaries:
        min_base, max_base = boundaries.pop(0)
        group = [min_base, max_base]

        i = 0
        while i < len(boundaries):
            min_check, max_check = boundaries[i]
            if np.all(min_check >= (min_base - margin)) and np.all(max_check <= (max_base + margin)):
                group.append(min_check)
                group.append(max_check)
                boundaries.pop(i)
            else:
                i += 1

        min_final = np.min(group, axis=0)
        max_final = np.max(group, axis=0)

        # Crear la bounding box correctamente con 8 puntos
        bounding_box = np.array([
            [min_final[0], min_final[1], min_final[2]],
            [min_final[0], min_final[1], max_final[2]],
            [min_final[0], max_final[1], min_final[2]],
            [min_final[0], max_final[1], max_final[2]],
            [max_final[0], min_final[1], min_final[2]],
            [max_final[0], min_final[1], max_final[2]],
            [max_final[0], max_final[1], min_final[2]],
            [max_final[0], max_final[1], max_final[2]]
        ])

        merged.append(bounding_box)

    return merged

#%%
def img_window_lst(volumne_img, corte):
    """
    Genera una lista de imágenes en escala de grises a partir de un volumen 3D,
    aplicando una ventana de valores de intensidad y normalización.

    Parámetros:
      - volumne_img: Array 3D que representa la imagen volumétrica.
      - corte: Índice del eje que define el tipo de corte:
          # 0: Corte axial
          # 1: Corte coronal
          # 2: Corte sagital

    Return:
      - Lista de tuplas (índice, imagen), donde:
          # índice: Número de la capa en la dirección del eje especificado.
          # imagen: Imagen en formato PIL normalizada y en escala de grises.
    """

    lst_final = []
    for i in range(volumne_img.shape[corte]):
        if corte == 0:
            img_w = np.clip(volumne_img[i, :, :], -150, 250)
            # Guardamos normalizada en uint8 en formato .png
            img_norm = (((img_w + 150) / (250 + 150)) * 255).astype(np.uint8)  # Normalizar con escala
            img_n = Image.fromarray(img_norm).convert('L')
            lst_final.append((i, img_n))

        elif corte == 1:
            img_w = np.clip(volumne_img[:, i, :], -150, 250)
            # Guardamos normalizada en uint8 en formato .png
            img_norm = (((img_w + 150) / (250 + 150)) * 255).astype(np.uint8)
            img_n = Image.fromarray(img_norm).convert('L')# Normalizar con escala
            lst_final.append((i, img_n))

        elif corte == 2:
            img_w = np.clip(volumne_img[:, :, i], -150, 250)
            # Guardamos normalizada en uint8 en formato .png
            img_norm = (((img_w + 150) / (250 + 150)) * 255).astype(np.uint8)  # Normalizar con escala
            img_n = Image.fromarray(img_norm).convert('L')
            lst_final.append((i, img_n))

    return lst_final

