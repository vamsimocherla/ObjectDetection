import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# import the necessary packages
from mrcnn.config import Config
from mrcnn import utils
from imutils import paths
import numpy as np
import argparse
import cv2
import os
import json
import skimage.draw
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import time

############################################################
#  Initializations
############################################################

# initialize the dataset path, images path, and annotations file path
ROOT_PATH = os.path.abspath("/home/vamsimocherla/Storage/DL4OD/mrcnn_models")
DATASET_PATH = os.path.sep.join([ROOT_PATH, "datasets/HomeObjects06"])
IMAGES_PATH = os.path.sep.join([DATASET_PATH, "Train"])
MASKS_PATH = os.path.sep.join([DATASET_PATH, "Masks"])
AUG_IMG_PATH = os.path.sep.join([DATASET_PATH, "AugmentedTrain"])
AUG_MASK_PATH = os.path.sep.join([DATASET_PATH, "AugmentedMasks"])
ANNOT_PATH = os.path.sep.join([DATASET_PATH, "home_objects_train.json"])
AUG_JSON_PATH = os.path.sep.join([DATASET_PATH, "home_objects_train_aug.json"])

# grab all image paths
IMAGE_PATHS = sorted(list(paths.list_images(IMAGES_PATH)))
idxs = list(range(0, len(IMAGE_PATHS)))

# initialize the class names dictionary
with open(ANNOT_PATH) as json_file:
    data = json.load(json_file)
    CLASS_NAMES = data['_via_attributes']['region']['type']['options']
    # print("[INFO] Class Names: {}".format(CLASS_NAMES))

# initialize the path to the Mask R-CNN pre-trained on COCO
COCO_PATH = "models/mask_rcnn_coco.h5"

# initialize the name of the directory where logs and output model
# snapshots will be stored
LOGS_AND_MODEL_DIR = "logs"

# to make the data uniform distribution
TOTAL_IMG_PER_CLASS = 100

############################################################
#  Configurations
############################################################


class ObjectsConfig(Config):
    # give the configuration a recognizable name
    NAME = "home_objects"

    # set the number of GPUs to use training along with the number of
    # images per GPU (which may have to be tuned depending on how
    # much memory your GPU has)
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # set the minimum detection confidence (used to prune out false
    # positive detections)
    DETECTION_MIN_CONFIDENCE = 0.9

    # set the number of steps per training epoch
    # STEPS_PER_EPOCH = len(trainIdxs) // (IMAGES_PER_GPU * GPU_COUNT)
    STEPS_PER_EPOCH = 100

    # number of classes (+1 for the background)
    NUM_CLASSES = len(CLASS_NAMES) + 1


class ObjectsInferenceConfig(ObjectsConfig):
    # set the number of GPUs and images per GPU (which may be
    # different values than the ones used for training)
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # set the minimum detection confidence (used to prune out false
    # positive detections)
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################


class ObjectsDataset(utils.Dataset):
    def __init__(self, imagesPath, annotPath, classNames, scaleWidth=800):
        # call the parent constructor
        super().__init__(self)

        # store the image paths and class names along with the width
        # we’ll resize images to
        self.imagesPath = imagesPath
        self.classNames = classNames
        self.scaleWidth = scaleWidth

        # load the annotation data
        self.annots = self.load_annotation_data(annotPath)

    def load_annotation_data(self, annotPath):
        # load the contents of the annotation JSON file (created
        # using the VIA tool) and initialize the annotations
        # dictionary
        annotations = json.loads(open(annotPath).read())['_via_img_metadata']
        annots = {}

        # loop over the file ID and annotations themselves (values)
        for (fileID, data) in sorted(annotations.items()):
            # store the data in the dictionary using the filename as
            # the key
            # print("[INFO] Reading File: {}".format(fileID))
            annots[data["filename"]] = data

        # return the annotations dictionary
        # print("[INFO] annots: {}".format(annots))
        return annots

    def load_objects(self, idxs):
        # loop over all class names and add each to the 'objects'
        # dataset
        classIndex = 1
        for (classID, label) in self.classNames.items():
            # print("[INFO] classID: {} label: {}".format(classID, label))
            self.add_class("objects", classIndex, label)
            classIndex += 1

        # loop over the image path indexes
        for i in idxs:
            # extract the image filename to serve as the unique image ID
            imagePath = self.imagesPath[i]
            filename = imagePath.split(os.path.sep)[-1]
            # print("[INFO] imagePath: {} filename: {}".format(imagePath, filename))

            # load the image and resize it so we can determine its
            # width and height (unfortunately VIA does not embed
            # this information directly in the annotation file)
            image = cv2.imread(imagePath)
            origH, origW = image.shape[:2]
            # image = imutils.resize(image, width=self.scaleWidth)
            (newH, newW) = image.shape[:2]

            # add the image to the dataset
            self.add_image("objects", image_id=filename,
                           width=newW, height=newH,
                           orig_width=origW, orig_height=origH,
                           path=imagePath)

    def load_image(self, imageID):
        # grab the image path, load it, and convert it from BGR to
        # RGB color channel ordering
        imagePath = self.image_info[imageID]["path"]
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # resize the image, preserving the aspect ratio
        # image = imutils.resize(image, width=self.scaleWidth)

        # return the image
        return image

    def load_mask(self, imageID):
        # grab the image info and then grab the annotation data for
        # the current image based on the unique ID
        info = self.image_info[imageID]
        annot = self.annots[info["id"]]
        classIDs = np.zeros(len(annot["regions"]))
        # print("[INFO] info => {}".format(info))
        # print("[INFO] annot => {}".format(annot))
        # print("[INFO] classIDs => {}".format(classIDs))

        # allocate memory for our [height, width, num_instances] array
        # where each "instance" effectively has its own "channel"
        masks = np.zeros((info["height"], info["width"],
                          len(annot["regions"])), dtype="uint8")
        # print("[INFO] masks.shape => {}".format(masks.shape))

        # loop over each of the annotated regions
        for (i, region) in enumerate(annot["regions"]):
            # grab the shape and region attributes
            sa = region["shape_attributes"]
            ra = region["region_attributes"]

            # add the corresponding classID
            classIDs[i] = int(ra['type'])

            # get the corresponding mask based on the region shape
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = self.get_mask(sa, info)
            # print("[INFO] rr: {} rr.shape: {} cc: {} cc.shape: {}".
            #       format(rr, rr.shape, cc, cc.shape))

            # Note that this modifies the existing array arr, instead of creating a result array
            # rr[rr > masks.shape[0] - 1] = masks.shape[0] - 1
            # cc[cc > masks.shape[1] - 1] = masks.shape[1] - 1

            masks[rr, cc, i] = 255

            display = False
            if display:
                cv2.imshow("input", cv2.imread(info['path']))
                cv2.imshow("regionMask", masks[:, :, i])
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                cv2.waitKey(1)

        # print("[INFO] classIDs => {}".format(classIDs))
        # return the mask array and class IDs
        return masks.astype("bool"), classIDs.astype("int32")

    def get_mask(self, sa, info):

        # scale the center (x, y)-coordinates and radius of the
        # circle based on the dimensions of the resized image
        # ratio = info["width"] / float(info["orig_width"])
        ratio = 1
        shape = (info["height"], info["width"])

        # handle mask shape
        if sa['name'] == 'circle':
            cX = int(sa["cx"] * ratio)
            cY = int(sa["cy"] * ratio)
            r = int(sa["r"] * ratio)
            rr, cc = skimage.draw.circle(cY, cX, r, shape)

        if sa['name'] == 'ellipse':
            cX = int(sa["cx"] * ratio)
            cY = int(sa["cy"] * ratio)
            rX = int(sa["rx"] * ratio)
            rY = int(sa["ry"] * ratio)
            theta = int(sa["theta"] * ratio)
            rr, cc = skimage.draw.ellipse(cY, cX, rX, rY,
                                          rotation=theta, shape=shape)

        if sa['name'] == 'polygon':
            pointsY = np.asarray(sa['all_points_y']) * ratio
            pointsX = np.asarray(sa['all_points_x']) * ratio
            # print("[INFO] sa_y: {} sa_x: {}".format(sa['all_points_y'], sa['all_points_x']))
            rr, cc = skimage.draw.polygon(pointsY.astype("int32"),
                                          pointsX.astype("int32"), shape)

        return rr, cc


def get_augmentation(augment):
    import imgaug.augmenters as iaa
    from keras.preprocessing.image import ImageDataGenerator
    augmentation = None
    if augment == 0:
        augmentation = iaa.Sometimes(.667, iaa.Sequential([
            iaa.Fliplr(0.5),  # horizontal flips
            iaa.Crop(percent=(0, 0.1)),  # random crops
            # Small gaussian blur with random sigma between 0 and 0.25.
            # But we only blur about 50% of all images.
            iaa.Sometimes(0.5,
                          iaa.GaussianBlur(sigma=(0, 0.25))
                          ),
            # Strengthen or weaken the contrast in each image.
            iaa.ContrastNormalization((0.75, 1.5)),
            # Add gaussian noise.
            # For 50% of all images, we sample the noise once per pixel.
            # For the other 50% of all images, we sample the noise per pixel AND
            # channel. This can change the color (not only brightness) of the
            # pixels.
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255)),
            # Make some images brighter and some darker.
            # In 20% of all cases, we sample the multiplier once per channel,
            # which can end up changing the color of the images.
            iaa.Multiply((0.8, 1.2)),
            # Apply affine transformations to each image.
            # Scale/zoom them, translate/move them, rotate them and shear them.
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-180, 180),
                # shear=(-8, 8)
            )
        ], random_order=True))  # apply augmenters in random order

    if augment == 1:
        # construct the image generator for data augmentation then
        # initialize the total number of images generated thus far
        augmentation = ImageDataGenerator(
            rotation_range=30,
            zoom_range=0.15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            horizontal_flip=True,
            fill_mode="nearest")

    return augmentation


def get_freq(dataset):
    freq = {}
    for value in CLASS_NAMES.values():
        freq[value] = 0
    for i in range(len(idxs)):
        (masks, classIDs) = dataset.load_mask(i)
        for classID in classIDs:
            freq[CLASS_NAMES[str(classID)]] += 1

    return freq


def augment_image(imagePath, maskPath):
    from keras.preprocessing.image import img_to_array
    from keras.preprocessing.image import load_img
    image = load_img(imagePath)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # get the augmentation
    aug = get_augmentation(1)
    total = 0

    print("[INFO] generating images...")
    imageGen = aug.flow(image, batch_size=1, save_to_dir=AUG_IMG_PATH,
                        save_prefix="aug_image", save_format="jpg")
    # loop over examples from our image data augmentation generator
    for image in imageGen:
        # increment our counter
        total += 1
        # if we have reached the specified number of examples, break
        # from the loop
        if total == 10:
            break

    total = 0
    maskImage = load_img(maskPath)
    maskImage = img_to_array(maskImage)
    maskImage = np.expand_dims(maskImage, axis=0)

    maskGen = aug.flow(maskImage, batch_size=1, save_to_dir=AUG_MASK_PATH,
                       save_prefix="aug_mask", save_format="jpg")
    # loop over examples from our image data augmentation generator
    for mask in maskGen:
        # increment our counter
        total += 1
        # if we have reached the specified number of examples, break
        # from the loop
        if total == 10:
            break


def get_sequential(seq_id=0):
    import imgaug as ia
    import imgaug.augmenters as iaa

    seq = None
    if seq_id == 0:
        seq = iaa.Sequential([
            iaa.Fliplr(0.5),  # horizontal flips
            iaa.Crop(percent=(0, 0.1)),  # random crops
            # Small gaussian blur with random sigma between 0 and 0.5.
            # But we only blur about 50% of all images.
            iaa.Sometimes(0.5,
                          iaa.GaussianBlur(sigma=(0, 0.5))
                          ),
            # Strengthen or weaken the contrast in each image.
            iaa.ContrastNormalization((0.75, 1.5)),
            # Add gaussian noise.
            # For 50% of all images, we sample the noise once per pixel.
            # For the other 50% of all images, we sample the noise per pixel AND
            # channel. This can change the color (not only brightness) of the
            # pixels.
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
            # Make some images brighter and some darker.
            # In 20% of all cases, we sample the multiplier once per channel,
            # which can end up changing the color of the images.
            iaa.Multiply((0.8, 1.2), per_channel=0.2),
            # Apply affine transformations to each image.
            # Scale/zoom them, translate/move them, rotate them and shear them.
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-25, 25),
                shear=(-8, 8)
            )
        ], random_order=True)  # apply augmenters in random order

    if seq_id == 1:
        seq = iaa.Sequential([
            iaa.Fliplr(0.5),  # horizontal flip 50% of all images
            iaa.Flipud(0.2),  # vertically flip 20% of all images
            # crop images by -5% to 10% of their height/width
            iaa.Sometimes(0.50, iaa.CropAndPad(
                percent=(-0.05, 0.1),
                pad_mode=ia.ALL,
                pad_cval=(0, 255)
            )),
            iaa.Sometimes(0.50, iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent (per axis)
                rotate=(-45, 45),  # rotate by -45 to +45 degrees
                shear=(-16, 16),  # shear by -16 to +16 degrees
                order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL  # use any of scikit-image's warping modes
            ))
        ], random_order=True)  # apply augmenters in random order

    if seq_id == 2:
        seq = iaa.Sequential([
            iaa.Dropout([0.05, 0.2]),  # drop 5% or 20% of all pixels
            iaa.Sharpen((0.0, 1.0)),  # sharpen the image
            iaa.Affine(rotate=(-45, 45)),  # rotate by -45 to 45 degrees (affects segmaps)
            iaa.ElasticTransformation(alpha=50, sigma=5)  # apply water effect (affects segmaps)
        ], random_order=True)

    if seq_id == 3:
        seq = iaa.Sequential([
            iaa.Fliplr(0.5),  # horizontal flip 50% of all images
            iaa.Flipud(0.2),  # vertically flip 20% of all images
            iaa.Multiply((0.5, 1.5), per_channel=0.5),  # Change brightness of images (50-150% of original value)
            iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),  # Improve or worsen the contrast of images
            iaa.Sometimes(0.50, iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # scale images to 80-120% of their size
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent (per axis)
                rotate=(-45, 45),  # rotate by -45 to +45 degrees
                shear=(-16, 16),  # shear by -16 to +16 degrees
                order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                # cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                # mode=ia.ALL  # use any of scikit-image's warping modes
            ))
        ], random_order=True)  # apply augmenters in random order

    return seq


def augment_data(image, mask, fileName, total, className):
    from imgaug.augmentables.segmaps import SegmentationMapOnImage

    segmap = SegmentationMapOnImage(mask, shape=mask.shape,
                                    nb_classes=1 + 1)

    # Define our augmentation pipeline.
    seq = get_sequential(seq_id=3)

    # save the original image and mask
    plt.imsave(os.path.sep.join([AUG_IMG_PATH, className, fileName]), image)
    plt.imsave(os.path.sep.join([AUG_MASK_PATH, className, fileName]),
               mask, cmap=cm.gray)

    # save the augmented images and segmaps
    fileName = os.path.splitext(fileName)[0]
    for i in range(total):
        image_aug, segmap_aug = seq(image=image, segmentation_maps=segmap)
        imagePath = "{}_{}.JPG".format(fileName, i)
        plt.imsave(os.path.sep.join([AUG_IMG_PATH, className, imagePath]), image_aug)
        maskPath = "{}_{}.JPG".format(fileName, i)
        segmap_aug = segmap_aug.draw(size=image_aug.shape[:2]).astype("bool")
        plt.imsave(os.path.sep.join([AUG_MASK_PATH, className, maskPath]),
                   segmap_aug[:, :, 0], cmap=cm.gray)


def create_directory(className):
    # print("[INFO] Creating Class Directory: {}".format(className))
    imgDirectory = os.path.sep.join([AUG_IMG_PATH, className])
    if not os.path.exists(imgDirectory):
        os.makedirs(imgDirectory)
    maskDirectory = os.path.sep.join([AUG_MASK_PATH, className])
    if not os.path.exists(maskDirectory):
        os.makedirs(maskDirectory)


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--mode", required=True,
                    help="either 'save_masks', 'img_gt'")
    ap.add_argument("-w", "--weights",
                    help="optional path to pretrained weights")
    ap.add_argument("-i", "--image",
                    help="optional path to input image to segment")
    args = vars(ap.parse_args())

    # load the entire dataset
    dataset = ObjectsDataset(IMAGE_PATHS, ANNOT_PATH, CLASS_NAMES)
    dataset.load_objects(idxs)
    dataset.prepare()
    print("[INFO] Data Loaded")

    if args["mode"] == "save_masks":
        for i in range(len(idxs)):
            image = dataset.load_image(i)
            imagePath = dataset.image_info[i]["path"]
            fileName = imagePath.split(os.path.sep)[-1]
            (masks, classIDs) = dataset.load_mask(i)
            print("[INFO] filename: {}".format(fileName))
            maskPath = os.path.sep.join([MASKS_PATH, fileName])
            plt.imsave(maskPath, masks[:, :, 0], cmap=cm.gray)

            display = False
            if display:
                plt.imshow(image)
                plt.imshow(masks[:, :, 0], cmap='gray')
                plt.show()

    if args["mode"] == "img_gt":
        # compute the frequency distribution
        freq = get_freq(dataset)

        start = time.time()
        augmentedData = {}
        for i in range(len(idxs)):
            image = dataset.load_image(i)
            imagePath = dataset.image_info[i]["path"]
            fileName = imagePath.split(os.path.sep)[-1]
            (masks, classIDs) = dataset.load_mask(i)
            className = CLASS_NAMES[str(classIDs[0])]
            # create a directory if not exists
            create_directory(className)
            print("[INFO] filename: {} class: {}".format(fileName,
                                                         className))
            # sequential map
            total = math.floor(TOTAL_IMG_PER_CLASS / freq[className])
            augment_data(image, masks[:, :, 0], fileName,
                         total, className)
            augData = {}
            augData["class_name"] = className
            augData["total"] = total

            augmentedData[fileName] = augData

        end = time.time()
        print("[INFO] Data Augmentation Time: {} seconds".format(end - start))
        # save in JSON format
        json_data = json.dumps(augmentedData)
        print(json_data)
