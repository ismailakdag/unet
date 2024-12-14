import os
path_images = os.path.join(r'D:\Monkeypox_Segmentation\dataset\dataset_v3\masks_3')
new_path_images = os.path.join(r'D:\Monkeypox_Segmentation\dataset\dataset_v3\masks_3')
images_list = os.listdir(path_images)
print(images_list)
for i, patient in enumerate(images_list):
  imagepath = os.path.join(path_images, patient)
  bol = patient.split(".")
  yeni_isim = bol[0] + "_mask." + bol[1]
  new_imagepath = os.path.join(new_path_images, yeni_isim)
  os.rename(imagepath,new_imagepath )
