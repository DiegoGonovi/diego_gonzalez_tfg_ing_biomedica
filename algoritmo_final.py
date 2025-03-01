import time
import numpy as np
from ultralytics import YOLO
from Funciones_fusion import (yolo_detections, find_vois, txt_array_gt, full_one_voi,
                                 axial_coronal_voi, axial_sagital_voi, coronal_sagital, gt_voi_in_txt_2,
                                 plot_2_vois_3_slices)


test_pacientes = ['054', '049', '051', '164', '078', '092', '010', '030',
                  '178', '068', '109', '166', '019', '163', '035', '123', '125']

model_path_axial = 'C:/Users/diego/Desktop/TFG/ProyectoTFG/004-RESULTS/Axial/yolov8x-axial_300/weights/best.pt'
model_axial = YOLO(model_path_axial)

model_path_coronal = ('C:/Users/diego/Desktop/TFG/ProyectoTFG/004-RESULTS/Coronal/'
                      'yolov8x-coronal_norm_300/weights/best.pt')
model_coronal = YOLO(model_path_coronal)

model_path_sagital = ('C:/Users/diego/Desktop/TFG/ProyectoTFG/004-RESULTS/Sagital/'
                      'yolov8x-sagital_norm_300/weights/best.pt')
model_sagital = YOLO(model_path_sagital)

output_path = 'C:/Users/diego/Desktop/TFG/ProyectoTFG/VOIs_predict'
output_dir = 'C:/Users/diego/Desktop/TFG/ProyectoTFG/GT_VOI'

start_time = time.time()
execution_times = []  # Lista para almacenar tiempos por paciente


# Generar una gráfica para cada paciente combinando superficies axiales, coronales y sagitales
for pat in sorted(test_pacientes):
    start_time_patient = time.time()

    path = f'C:/Users/diego/Desktop/TFG/ProyectoTFG/Pacientes_Yolo/{pat}'
    path_a, path_c, path_s = f"{path}/YOLO/Axial/Images", f"{path}/YOLO/Coronal/Images", f"{path}/YOLO/Sagital/Images"

    rois_a, rois_c, rois_s = (yolo_detections(path_a, model_axial, 'axial'),
                              yolo_detections(path_c, model_coronal, 'coronal'),
                              yolo_detections(path_s, model_sagital, 'sagital'))

    full_vois_3d = find_vois(rois_s, rois_c, rois_a)
    gt_voi = txt_array_gt(output_dir, pat, 8, 3)

    if len(full_vois_3d) < 1:
        vois_2d = []
        vois_axial_coronal = axial_coronal_voi(rois_c, rois_a)
        if len(vois_axial_coronal) >= 1:
            onevoi_ac = full_one_voi(vois_axial_coronal)
            vois_2d.append(onevoi_ac)

        vois_axial_sagital = axial_sagital_voi(rois_s, rois_a)
        if len(vois_axial_sagital) >= 1:
            onevoi_as = full_one_voi(vois_axial_sagital)
            vois_2d.append(onevoi_as)

        vois_coronal_sagital = coronal_sagital(rois_c, rois_s)
        if len(vois_coronal_sagital) >= 1:
            onevoi_cs = full_one_voi(vois_coronal_sagital)
            vois_2d.append(onevoi_cs)

        if len(vois_2d) >= 1:
            plot_2_vois_3_slices(pat, rois_a, rois_c, rois_s, vois_2d, gt_voi)
            gt_voi_in_txt_2(vois_2d, output_path, str(pat + '_predict'))

        if len(vois_2d) < 1:
            plot_2_vois_3_slices(pat, rois_a, rois_c, rois_s, full_vois_3d, gt_voi)
            gt_voi_in_txt_2(full_vois_3d, output_path, str(pat + '_predict'))

    else:
        one_v = full_one_voi(full_vois_3d)
        plot_2_vois_3_slices(pat, rois_a, rois_c, rois_s, [one_v], gt_voi)
        gt_voi_in_txt_2([one_v], output_path, str(pat + '_predict'))

    end_time_patient = time.time()
    execution_time_patient = end_time_patient - start_time_patient
    execution_times.append(execution_time_patient)  # Guardar tiempo individual
    print(f"Tiempo de ejecución para el paciente {pat}: {execution_time_patient:.2f} segundos")

    #input("Presiona Enter para ver el siguiente paciente...")
    #time.sleep(4)
    break

end_time = time.time()
total_execution_time = end_time - start_time

mean_execution_time = np.mean(execution_times)
std_execution_time = np.std(execution_times)

# Mostrar resultados
print(f"\nTiempo total de ejecución: {total_execution_time:.2f} segundos")
print(f"Tiempo medio por paciente: {mean_execution_time:.2f} ± {std_execution_time:.2f} segundos")

