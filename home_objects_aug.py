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
import re
import time
import logging

############################################################
#  Initializations
############################################################

# initialize the dataset path, images path, and annotations file path
ROOT_PATH = os.path.abspath("/home/vamsimocherla/Storage/DL4OD/mrcnn_models")
DATASET_PATH = os.path.sep.join([ROOT_PATH, "datasets/HomeObjects06"])
AUG_JSON_PATH = os.path.sep.join([DATASET_PATH, "home_objects_train_aug.json"])
MASKS_PATH = os.path.sep.join([DATASET_PATH, "AugmentedMasks"])
OUTPUT_PATH = os.path.sep.join([DATASET_PATH, "Output"])

# list of new classes to train on
ALL_CLASSES = ["adhesive tape", "aluminium foil", "backpack", "bike gear", "bike handle",
           "bike part", "bike wheel", "blender", "blender blade", "book", "bottle",
           "brass", "bread", "bread packet", "bubble wrap", "cd rack", "cellphone",
           "chair", "clock", "cloth", "couch", "crusher", "cube", "cup", "fish", "glove",
           "handwash", "headphones", "ice tray", "key holder", "lemon", "measuring tape",
           "miscellaneous", "multi-tool", "noodles", "paper bag", "plastic bag", "pot",
           "potted plant", "pouch", "remote", "rice bag", "rice cooker", "scissors",
           "scotch brite", "scrub brush", "spin tool", "sponge pipe", "sunglasses", "vacuum"]

# if training on a subset of classes
# CLASSES = ["adhesive tape", "aluminium foil", "backpack", "bike gear", "bike handle",
#            "bike part", "bike wheel", "blender blade", "book", "bottle"]
# CLASSES = ["brass", "bread", "bread packet", "bubble wrap", "cd rack", "cellphone",
#            "chair", "clock", "cloth", "couch", "crusher", "cube", "cup", "fish", "glove"]
# CLASSES = ["handwash", "headphones", "ice tray", "key holder", "lemon", "measuring tape",
#            "miscellaneous", "multi-tool", "noodles", "paper bag", "plastic bag", "pot"]
# CLASSES = ["potted plant", "pouch", "remote", "rice bag", "rice cooker", "scissors",
#            "scotch brite", "scrub brush", "spin tool", "sponge pipe", "sunglasses", "vacuum"]
# if training on all classes
CLASSES = ALL_CLASSES
CLASS_NAMES = {x: (i+1) for i, x in enumerate(CLASSES)}
IMAGES_PATH = []
for c in CLASSES:
    imagesPath = os.path.sep.join([DATASET_PATH, "AugmentedTrain", c])
    IMAGES_PATH.extend(sorted(list(paths.list_images(imagesPath))))

# initialize the amount of data to use for training
TRAINING_SPLIT = 0.75

# grab all image paths, then randomly select indexes for both training
# and validation
total_idxs = list(range(0, len(IMAGES_PATH)))
random.seed(42)
random.shuffle(total_idxs)
i = int(len(total_idxs) * TRAINING_SPLIT)
trainIdxs = total_idxs[:i]
valIdxs = total_idxs[i:]
# print("[INFO] idxs: {}".format(len(total_idxs)))
# print("[INFO] trainIdxs: {}".format(len(trainIdxs)))
# print("[INFO] valIdxs: {}".format(len(valIdxs)))

# initialize the class names dictionary
with open(AUG_JSON_PATH) as json_file:
    AUG_JSON = json.load(json_file)
    # CLASS_NAMES = set()
    # for key, value in AUG_JSON.items():
    #     CLASS_NAMES.add(value["class_name"])
    # CLASS_NAMES = {x: (i+1) for i, x in enumerate(CLASS_NAMES)}
    # print(CLASS_NAMES)

# initialize the path to the Mask R-CNN pre-trained on COCO
COCO_PATH = "models/mask_rcnn_coco.h5"

# initialize the name of the directory where logs and output model
# snapshots will be stored
LOGS_AND_MODEL_DIR = os.path.abspath("/home/vamsimocherla/Storage/DL4OD/mrcnn_models/logs")

############################################################
#  Configurations
############################################################


class ObjectsConfig(Config):
    # give the configuration a recognizable name
    NAME = "home_objects_all_classes_adam_"

    # Debug Logs
    LOG_FILE = os.path.sep.join([LOGS_AND_MODEL_DIR, NAME + "_debug_log.txt"])
    logging.basicConfig(filename=LOG_FILE, level=logging.DEBUG)

    # set the number of GPUs to use training along with the number of
    # images per GPU (which may have to be tuned depending on how
    # much memory your GPU has)
    GPU_COUNT = 2
    IMAGES_PER_GPU = 1

    # set the minimum detection confidence (used to prune out false
    # positive detections)
    DETECTION_MIN_CONFIDENCE = 0.9

    ### Task A ###
    # Backbone network architecture
    # Supported values are: resnet50, resnet101.
    # You can also provide a callable that should have the signature
    # of model.resnet_graph. If you do so, you need to supply a callable
    # to COMPUTE_BACKBONE_SHAPE as well
    BACKBONE = "resnet50"

    # set the number of steps per training epoch
    # STEPS_PER_EPOCH = len(trainIdxs) // (IMAGES_PER_GPU * GPU_COUNT)
    STEPS_PER_EPOCH = 300
    VALIDATION_STEPS = 50
    LEARNING_RATE = 0.001
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    # TRAIN_ROIS_PER_IMAGE = 512

    # number of classes (+1 for the background)
    NUM_CLASSES = len(CLASS_NAMES) + 1

    ### Task B ###
    # Loss weights for more precise optimization.
    # Can be used for R-CNN training setup.
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.0,
        "rpn_bbox_loss": 1.0,  # this has the top1 most affect on validation/unseen samples
        "mrcnn_class_loss": 1.0,
        "mrcnn_bbox_loss": 1.0,  # this has the top2 most affect on validation/unseen samples
        "mrcnn_mask_loss": 1.0  # this has the top3 most affect on validation/unseen samples
    }

    ### Task C ###
    # change optimizer to Adam
    OPTIMIZER = "ADAM"  # default is SGD

    ### Task D ###
    # test time augmentation


class ObjectsInferenceConfig(ObjectsConfig):
    # set the number of GPUs and images per GPU (which may be
    # different values than the ones used for training)
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Backbone network architecture
    # Supported values are: resnet50, resnet101.
    # You can also provide a callable that should have the signature
    # of model.resnet_graph. If you do so, you need to supply a callable
    # to COMPUTE_BACKBONE_SHAPE as well
    BACKBONE = "resnet50"

    # set the minimum detection confidence (used to prune out false
    # positive detections)
    DETECTION_MIN_CONFIDENCE = 0.5


############################################################
#  Dataset
############################################################


class ObjectsDataset(utils.Dataset):
    def __init__(self, imagesPath, masksPath, augJson,
                 classNames, scaleWidth=800):
        # call the parent constructor
        super().__init__(self)

        # store the image paths and class names
        self.imagesPath = imagesPath
        self.masksPath = masksPath
        self.augJson = augJson
        self.classNames = classNames
        self.scaleWidth = scaleWidth

    def load_objects(self, idxs):
        # loop over all class names and add each to the 'objects'
        # dataset
        classIndex = 1
        for (className, classID) in self.classNames.items():
            self.add_class("objects", classIndex, className)
            classIndex += 1

        # loop over the image path indexes
        for i in idxs:
            # extract the image fileName to serve as the unique image ID
            imagePath = self.imagesPath[i]
            fileName = imagePath.split(os.path.sep)[-1]
            # print("[INFO] loading file: {}".format(fileName))
            self.add_image_util(imagePath, fileName)

    def add_image_util(self, imagePath, fileName):
        image = cv2.imread(imagePath)
        height, width = image.shape[:2]
        # add the image to the dataset
        self.add_image("objects", image_id=fileName,
                       width=width, height=height,
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
        fileName = info["id"]
        # handle augmented fileNames
        className = AUG_JSON[re.sub(r'_[0-9]+', '', fileName)]["class_name"]
        classIDs = np.zeros(1)
        classIDs[0] = CLASS_NAMES[className]

        # allocate memory for our [height, width, num_instances] array
        # where each "instance" effectively has its own "channel"
        masks = np.zeros((info["height"], info["width"], 1), dtype="uint8")
        maskPath = os.path.sep.join([MASKS_PATH, className, fileName])
        # print("[INFO] fileName: {} maskPath: {}".format(fileName, maskPath))
        masks[:, :, 0] = skimage.io.imread(maskPath, as_gray=True)

        # return the mask array and class IDs
        return masks.astype("bool"), classIDs.astype("int32")

    def get_class_name(self, imageID):
        # grab the image info and then grab the annotation data for
        # the current image based on the unique ID
        info = self.image_info[imageID]
        fileName = info["id"]
        # handle augmented fileNames
        className = AUG_JSON[re.sub(r'_[0-9]+', '', fileName)]["class_name"]
        return className


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


def create_directory(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


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
        start = time.time()
        trainDataset = ObjectsDataset(IMAGES_PATH, MASKS_PATH,
                                      AUG_JSON, CLASS_NAMES)
        trainDataset.load_objects(trainIdxs)
        trainDataset.prepare()
        end = time.time()
        print("[INFO] Train Data Init Time: {} seconds".format(end-start))
        logging.info("Train Data Init Time: {} seconds".format(end-start))

        # load the validation dataset
        start = time.time()
        valDataset = ObjectsDataset(IMAGES_PATH, MASKS_PATH,
                                    AUG_JSON, CLASS_NAMES)
        valDataset.load_objects(valIdxs)
        valDataset.prepare()
        end = time.time()
        print("[INFO] Val Data Init Time: {} seconds".format(end-start))
        logging.info("Val Data Init Time: {} seconds".format(end-start))

        # initialize the training configuration
        config = ObjectsConfig()
        config.display()

        # initialize the model and load the COCO weights so we can
        # perform fine-tuning
        start = time.time()
        print("[INFO] Initializing model")
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=LOGS_AND_MODEL_DIR)

        # load the weights if provided
        if args["weights"] == "last":
            print("[INFO] Loading last saved weights from disk")
            model.load_weights(model.find_last(), by_name=True)
        else:
            print("[INFO] Loading weights from pre-trained COCO")
            model.load_weights(COCO_PATH, by_name=True,
                               exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                        "mrcnn_bbox", "mrcnn_mask"])
        end = time.time()
        print("[INFO] Model Init Time: {} seconds".format(end-start))
        logging.info("Model Init Time: {} seconds".format(end-start))

        augment = -1
        augmentation = get_augmentation(augment)
        # Since we're using a very small dataset, and starting from
        # COCO trained weights, we don't need to train too long. Also,
        # no need to train all layers, just the heads should do it.

        # train *just* the layer heads
        total_start = time.time()
        start = time.time()
        model.train(trainDataset, valDataset, epochs=20,
                    augmentation=augmentation,
                    layers="heads", learning_rate=config.LEARNING_RATE/10)
        end = time.time()
        print("[INFO] Training Time Phase 1: {} seconds".format(end-start))
        logging.info("Training Time Phase 1: {} seconds".format(end-start))

        # train *just* the layer heads
        start = time.time()
        model.train(trainDataset, valDataset, epochs=40,
                    augmentation=augmentation,
                    layers="heads", learning_rate=config.LEARNING_RATE/100)
        end = time.time()
        print("[INFO] Training Time Phase 2: {} seconds".format(end-start))
        logging.info("Training Time Phase 2: {} seconds".format(end-start))

        # unfreeze the body of the network and train *all* layers
        start = time.time()
        model.train(trainDataset, valDataset, epochs=60,
                    layers="all", learning_rate=config.LEARNING_RATE/100)
        end = time.time()
        print("[INFO] Training Time Phase 3: {} seconds".format(end-start))
        logging.info("Training Time Phase 3: {} seconds".format(end-start))
        total_end = time.time()
        print("[INFO] Total Training Time: {} seconds".format(total_end-total_start))
        logging.info("Total Training Time: {} seconds".format(total_end-total_start))

        '''
        total_start = time.time()
        start = time.time()
        model.train(trainDataset, valDataset, epochs=120,
                    augmentation=augmentation,
                    layers="heads", learning_rate=config.LEARNING_RATE / 10)
        end = time.time()
        print("[INFO] Training Time Phase 1: {} seconds".format(end-start))
        logging.info("Training Time Phase 1: {} seconds".format(end-start))

        start = time.time()
        model.train(trainDataset, valDataset, epochs=180,
                    augmentation=augmentation,
                    layers="heads", learning_rate=config.LEARNING_RATE / 100)
        end = time.time()
        print("[INFO] Training Time Phase 2: {} seconds".format(end-start))
        logging.info("Training Time Phase 2: {} seconds".format(end-start))

        # unfreeze the body of the network and train *all* layers
        start = time.time()
        model.train(trainDataset, valDataset, epochs=210,
                    layers="all", learning_rate=config.LEARNING_RATE / 100)
        end = time.time()
        print("[INFO] Training Time Phase 3: {} seconds".format(end-start))
        logging.info("Training Time Phase 3: {} seconds".format(end-start))
        total_end = time.time()
        print("[INFO] Total Training Time: {} seconds".format(total_end-total_start))
        logging.info("Total Training Time: {} seconds".format(total_end-total_start))
        '''

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
        print("[INFO] model_weights: {}".format(weights))
        model.load_weights(weights, by_name=True)

        # load the input image, convert it from BGR to RGB channel
        # ordering, and resize the image
        image = cv2.imread(args["image"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = skimage.io.imread(args["image"])

        # perform a forward pass of the network to obtain the results
        start = time.time()
        r = model.detect([image], verbose=1)[0]
        end = time.time()
        print("[INFO] Prediction Time: {} seconds".format(end-start))
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
            classes = dict([(value, key) for key, value in CLASS_NAMES.items()])
            label = classes[classID]
            score = r["scores"][i]

            # draw the class label and score on the image
            text = "{}: {:.4f}".format(label, score)
            print("[INFO] predictions: {}".format(text))
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(image, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # save the output to disk
        fileName = args["image"].split(os.path.sep)[-1]
        # plt.imsave(os.path.sep.join([OUTPUT_PATH, fileName]), image)
        # save in the corresponding model's path
        outputPath = weights.split(os.path.sep)[-2]
        outputPath = os.path.sep.join([OUTPUT_PATH, outputPath])
        create_directory(outputPath)
        cv2.imwrite(os.path.sep.join([outputPath, fileName]), image)

        # resize the image so it more easily fits on our screen
        image = imutils.resize(image, width=512)

        # show the output image
        cv2.imshow("Output", image)
        cv2.waitKey(0)

    # check to see if we are investigating our images and masks
    elif args["mode"] == "investigate":
        # load the training dataset
        trainDataset = ObjectsDataset(IMAGES_PATH, MASKS_PATH,
                                      AUG_JSON, CLASS_NAMES)
        trainDataset.load_objects(trainIdxs)
        trainDataset.prepare()
        print("[INFO] Training Data Loaded")

        # load the validation dataset
        valDataset = ObjectsDataset(IMAGES_PATH, MASKS_PATH,
                                    AUG_JSON, CLASS_NAMES)
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

        verify_random = True
        if verify_random:
            image_id = 0
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
        verify_masks = True
        if verify_masks:
            for i in np.random.choice(trainDataset.image_ids, 12):
                # load the image and masks for the sampled image
                print("[INFO] investigating image index: {}".format(i))
                print("[INFO] path: {}".format(trainDataset.image_info[i]["path"]))
                image = trainDataset.load_image(i)
                (masks, classIDs) = trainDataset.load_mask(i)
                print("[INFO] class IDs: {}".format(classIDs))

                # visualize the masks for the current image
                visualize.display_top_masks(image, masks, classIDs,
                                            trainDataset.class_names)

        verify_dist = False
        if verify_dist:
            freq = {}
            for key in CLASS_NAMES.keys():
                freq[key] = 0
            for trainId in range(len(trainIdxs)):
                className = trainDataset.get_class_name(trainId)
                freq[className] += 1
            for valId in range(len(valIdxs)):
                className = valDataset.get_class_name(valId)
                freq[className] += 1

            for key, value in freq.items():
                print("[INFO] {} {}".format(key, value))

