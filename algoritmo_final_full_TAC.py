from ultralytics import YOLO
import nibabel as nib
import numpy as np
import time
from Funciones_fusion import (yolo_detections_full, find_vois, axial_coronal_voi,
                              axial_sagital_voi, coronal_sagital, merge_vol,
                              img_window_lst, txt_array_gt, plot_2_vois_3_slices,
                              plot_voi_3_slices, gt_voi_in_txt_2)


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

for pat in sorted(test_pacientes):
    start_time_patient = time.time()

    path = f'C:/Users/diego/Desktop/TFG/ProyectoTFG/Pacientes_Yolo/{pat}'

    full_img = path + '/image_resampled.nii.gz'  # Selección de la full img
    img_vol = nib.load(full_img)
    img_vol_data = img_vol.get_fdata()  # Extraemos intensidades

    path_a, path_c, path_s = (img_window_lst(img_vol_data, 2), img_window_lst(img_vol_data, 1),
                              img_window_lst(img_vol_data, 0))

    rois_a, rois_c, rois_s = (yolo_detections_full(path_a, model_axial, 'axial'),
                              yolo_detections_full(path_c, model_coronal, 'coronal'),
                              yolo_detections_full(path_s, model_sagital, 'sagital'))

    full_vois_3d = find_vois(rois_s, rois_c, rois_a)
    gt_voi = txt_array_gt(output_dir, pat, 8, 3)
    print(pat)
    if full_vois_3d:
        merge_v = merge_vol(full_vois_3d, 10)
        print('Tipo I')
        plot_2_vois_3_slices(pat, rois_a, rois_c, rois_s, merge_v, gt_voi)
        gt_voi_in_txt_2(merge_v, output_path, str(pat + '_predict'))

    else:
        vois_2d = []
        vois_axial_coronal = axial_coronal_voi(rois_c, rois_a)
        if len(vois_axial_coronal) > 1:
            merge_v_ac = merge_vol(vois_axial_coronal, 10)
            vois_2d.append(merge_v_ac)

        vois_axial_sagital = axial_sagital_voi(rois_s, rois_a)
        if len(vois_axial_sagital) > 1:
            merge_v_as = merge_vol(vois_axial_sagital, 10)
            vois_2d.append(merge_v_as)

        vois_coronal_sagital = coronal_sagital(rois_c, rois_s)
        if len(vois_coronal_sagital) > 1:
            merge_v_cs = merge_vol(vois_coronal_sagital, 10)
            vois_2d.append(merge_v_cs)

        if len(vois_2d) >= 1:
            print('Tipo II')
            vois_2d_flat = []
            for item in vois_2d:
                if isinstance(item, list):
                    vois_2d_flat.extend(item)  # Desempaquetar si hay listas internas
                else:
                    vois_2d_flat.append(item)
            plot_2_vois_3_slices(pat, rois_a, rois_c, rois_s, vois_2d_flat,
                                 gt_voi)
            gt_voi_in_txt_2(vois_2d_flat, output_path, str(pat + '_predict'))

        else:
            print('Tipo III')
            plot_voi_3_slices(pat, rois_a, rois_c, rois_s, [gt_voi])
            gt_voi_in_txt_2(full_vois_3d, output_path, str(pat + '_predict'))

    end_time_patient = time.time()
    execution_time_patient = end_time_patient - start_time_patient
    execution_times.append(execution_time_patient)  # Guardar tiempo individual
    print(f"Tiempo de ejecución para el paciente {pat}: {execution_time_patient:.2f} segundos")

end_time = time.time()
total_execution_time = end_time - start_time

mean_execution_time = np.mean(execution_times)
std_execution_time = np.std(execution_times)

# Mostrar resultados
print(f"\nTiempo total de ejecución: {total_execution_time:.2f} segundos")
print(f"Tiempo medio por paciente: {mean_execution_time:.2f} ± {std_execution_time:.2f} segundos")



