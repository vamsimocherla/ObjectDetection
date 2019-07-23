import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# import the necessary packages
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
from mrcnn import utils
from imutils import paths
import numpy as np
import argparse
import imutils
import random
import cv2
import os
import json
import skimage.draw
import imgaug.augmenters as iaa

############################################################
#  Initializations
############################################################

# initialize the dataset path, images path, and annotations file path
DATASET_PATH = os.path.abspath("datasets/HomeObjects06")
IMAGES_PATH = os.path.sep.join([DATASET_PATH, "Train"])
ANNOT_PATH = os.path.sep.join([DATASET_PATH, "home_objects_train.json"])

# initialize the amount of data to use for training
TRAINING_SPLIT = 0.75

# grab all image paths, then randomly select indexes for both training
# and validation
IMAGE_PATHS = sorted(list(paths.list_images(IMAGES_PATH)))
idxs = list(range(0, len(IMAGE_PATHS)))
random.seed(42)
random.shuffle(idxs)
i = int(len(idxs) * TRAINING_SPLIT)
trainIdxs = idxs[:i]
valIdxs = idxs[i:]

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
    return augmentation


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--mode", required=True,
                    help="either 'train', 'predict', or 'investigate'")
    ap.add_argument("-w", "--weights",
                    help="optional path to pretrained weights")
    ap.add_argument("-i", "--image",
                    help="optional path to input image to segment")
    args = vars(ap.parse_args())

    # check to see if we are training the Mask R-CNN
    if args["mode"] == "train":
        # load the training dataset
        trainDataset = ObjectsDataset(IMAGE_PATHS, ANNOT_PATH, CLASS_NAMES)
        trainDataset.load_objects(trainIdxs)
        trainDataset.prepare()

        # load the validation dataset
        valDataset = ObjectsDataset(IMAGE_PATHS, ANNOT_PATH, CLASS_NAMES)
        valDataset.load_objects(valIdxs)
        valDataset.prepare()

        # initialize the training configuration
        config = ObjectsConfig()
        config.display()

        # initialize the model and load the COCO weights so we can
        # perform fine-tuning
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=LOGS_AND_MODEL_DIR)
        model.load_weights(COCO_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])

        augment = -1
        augmentation = get_augmentation(augment)
        # Since we're using a very small dataset, and starting from
        # COCO trained weights, we don't need to train too long. Also,
        # no need to train all layers, just the heads should do it.
        # train *just* the layer heads
        model.train(trainDataset, valDataset, epochs=10, augmentation=augmentation,
                    layers="heads", learning_rate=config.LEARNING_RATE)

        # unfreeze the body of the network and train *all* layers
        # model.train(trainDataset, valDataset, epochs=20,
        #             layers="all", learning_rate=config.LEARNING_RATE / 10)

    # check to see if we are predicting using a trained Mask R-CNN
    elif args["mode"] == "predict":
        # initialize the inference configuration
        config = ObjectsInferenceConfig()

        # initialize the Mask R-CNN model for inference
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=LOGS_AND_MODEL_DIR)

        # load our trained Mask R-CNN
        weights = args["weights"] if args["weights"] \
            else model.find_last()
        model.load_weights(weights, by_name=True)

        # load the input image, convert it from BGR to RGB channel
        # ordering, and resize the image
        image = cv2.imread(args["image"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # perform a forward pass of the network to obtain the results
        r = model.detect([image], verbose=1)[0]
        # print(r)

        # loop over of the detected object's bounding boxes and
        # masks, drawing each as we go along
        for i in range(0, r["rois"].shape[0]):
            mask = r["masks"][:, :, i]
            image = visualize.apply_mask(image, mask,
                                         (1.0, 0.0, 0.0), alpha=0.5)
            image = visualize.draw_box(image, r["rois"][i],
                                       (1.0, 0.0, 0.0))

        # convert the image back to BGR so we can use OpenCV's
        # drawing functions
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # loop over the predicted scores and class labels
        for i in range(0, len(r["scores"])):
            # extract the bounding box information, class ID, label,
            # and predicted probability from the results
            (startY, startX, endY, end) = r["rois"][i]
            classID = r["class_ids"][i]
            label = CLASS_NAMES[str(classID)]
            score = r["scores"][i]

            # draw the class label and score on the image
            text = "{}: {:.4f}".format(label, score)
            print("[INFO] predictions: {}".format(text))
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(image, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # resize the image so it more easily fits on our screen
        image = imutils.resize(image, width=512)

        # show the output image
        cv2.imshow("Output", image)
        cv2.waitKey(0)

    # check to see if we are investigating our images and masks
    elif args["mode"] == "investigate":
        # load the training dataset
        trainDataset = ObjectsDataset(IMAGE_PATHS, ANNOT_PATH, CLASS_NAMES)
        trainDataset.load_objects(trainIdxs)
        trainDataset.prepare()
        print("[INFO] Training Data Loaded")

        # load the validation dataset
        valDataset = ObjectsDataset(IMAGE_PATHS, ANNOT_PATH, CLASS_NAMES)
        valDataset.load_objects(valIdxs)
        valDataset.prepare()
        print("[INFO] Validation Data Loaded")

        # load the 0-th training image and corresponding masks and
        # class IDs in the masks
        display = False
        if display:
            for i in range(0, 20):
                image = trainDataset.load_image(i)
                (masks, classIDs) = trainDataset.load_mask(i)

                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (20, 500)
                fontScale = 2
                fontColor = (0, 0, 0)
                lineType = 2

                cv2.putText(image, "classIDs: {}".format(classIDs),
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor,
                            lineType)

                # cv2.imshow("input", image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # cv2.waitKey(1)

        verify_random = False
        if verify_random:
            image_id = 74
            image = trainDataset.load_image(image_id)
            print("[INFO] path: {}".format(trainDataset.image_info[image_id]["path"]))
            (masks, classIDs) = trainDataset.load_mask(image_id)
            # show the image spatial dimensions which is HxWxC
            print("[INFO] Image Shape: {}".format(image.shape))

            # show the masks shape which should have the same width and
            # height of the images but the third dimension should be
            # equal to the total number of instances in the image itself
            print("[INFO] Masks Shape: {}".format(masks.shape))

            # show the length of the class IDs list along with the values
            # inside the list -- the length of the list should be equal
            # to the number of instances dimension in the 'masks' array
            print("[INFO] class IDs length: {}".format(len(classIDs)))
            print("[INFO] class IDs: {}".format(classIDs))

        # determine a sample of training image indexes and loop over
        # them
        verify_masks = False
        if verify_masks:
            for i in np.random.choice(trainDataset.image_ids, 3):
                # load the image and masks for the sampled image
                print("[INFO] investigating image index: {}".format(i))
                print("[INFO] path: {}".format(trainDataset.image_info[i]["path"]))
                image = trainDataset.load_image(i)
                (masks, classIDs) = trainDataset.load_mask(i)
                print("[INFO] class IDs: {}".format(classIDs))

                for classID in classIDs:
                    print("[INFO] class ID: {}".format(CLASS_NAMES[str(classID)]))

                # visualize the masks for the current image
                visualize.display_top_masks(image, masks, classIDs,
                                            trainDataset.class_names)

        verify_dist = True
        if verify_dist:
            freq = {}
            for value in CLASS_NAMES.values():
                freq[value] = 0
            for i in range(len(trainIdxs)):
                (masks, classIDs) = trainDataset.load_mask(i)
                for classID in classIDs:
                    freq[CLASS_NAMES[str(classID)]] += 1
            for i in range(len(valIdxs)):
                (masks, classIDs) = valDataset.load_mask(i)
                for classID in classIDs:
                    freq[CLASS_NAMES[str(classID)]] += 1

            for key, value in freq.items():
                print("[INFO] {} {}".format(key, value))

