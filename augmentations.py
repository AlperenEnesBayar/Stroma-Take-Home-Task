import os.path

import cv2
from glob import glob
import albumentations as A
import numpy as np

# Seperate labels from class and returns (/w expanded dim)
def LabelSeperator(org_label):
    if len(org_label.shape) == 1:
        seperated_label = [org_label[1:]]
        seperated_class = [org_label[:1]]
    else:
        seperated_label = []
        seperated_class = []
        for s_label in org_label:
            seperated_label.append(s_label[1:])
            seperated_class.append(s_label[:1])
    return seperated_label, seperated_class


# Augment and save the data and its label the specified number of times
def ImageAugmentation(image_path, count=1):  # Ex: "bnn_data\\images\\*\\*"
    image_paths = glob(image_path)

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Blur(blur_limit=2, p=0.2),
        A.GaussianBlur(blur_limit=(3, 3), sigma_limit=0, p=0.2),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.2),
        A.MedianBlur(blur_limit=3, p=0.2),
        A.FancyPCA(alpha=0.1, p=0.2),
        A.ShiftScaleRotate(p=0.2)
    ])

    transform_w_bb = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Blur(blur_limit=2, p=0.2),
        A.GaussianBlur(blur_limit=(3, 3), sigma_limit=0, p=0.2),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.2),
        A.MedianBlur(blur_limit=3, p=0.2),
        A.FancyPCA(alpha=0.1, p=0.2),
        A.ShiftScaleRotate(p=0.2)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    for image_index, image_path in enumerate(image_paths):
        print(f'\r Current image: {image_index + 1}/{len(image_paths)} Image_path: {image_path}', end='')
        s_img_path = image_path
        label_path = image_path.replace('images', 'labels').replace('.jpg', '.txt')

        # Check if label path exist, if not just augment image
        if os.path.exists(label_path):
            org_img = cv2.imread(s_img_path)
            org_label = np.loadtxt(label_path)

            seperated_label, seperated_class = LabelSeperator(org_label)

            for augment_counter in range(count):
                transformed = transform_w_bb(image=org_img, bboxes=seperated_label, class_labels=seperated_class)
                transformed_image = transformed['image']
                transformed_bboxes = transformed['bboxes']
                transformed_class_labels = transformed['class_labels']

                cv2.imwrite(
                    image_path.replace('bnn_data', 'bnn_data_aug').replace('.jpg', str(augment_counter) + '.jpg'),
                    transformed_image)
                aug_label_path = label_path.replace('bnn_data', 'bnn_data_aug').replace('.txt',
                                                                                        str(augment_counter) + '.txt')
                f = open(aug_label_path, "a")
                for b_index in range(len(transformed_bboxes)):
                    f.write(str(transformed_class_labels[b_index][0]) + " " +
                            str(transformed_bboxes[b_index][0]) + " " +
                            str(transformed_bboxes[b_index][1]) + " " +
                            str(transformed_bboxes[b_index][2]) + " " +
                            str(transformed_bboxes[b_index][3]) + '\n')
                f.close()

        else:
            org_img = cv2.imread(s_img_path)
            for augment_counter in range(count):
                transformed = transform(image=org_img)
                transformed_image = transformed['image']

                cv2.imwrite(
                    image_path.replace('bnn_data', 'bnn_data_aug').replace('.jpg', str(augment_counter) + '.jpg'),
                    transformed_image)


ImageAugmentation("bnn_data\\images\\train\\*", 10)
