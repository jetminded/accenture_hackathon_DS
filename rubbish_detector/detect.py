import cv2
from skimage.color import rgb2hsv, label2rgb
from skimage.filters import threshold_otsu
from skimage.morphology import closing, square
from skimage.measure import label, regionprops
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


class DetectRubbish:
    def __init__(self, threshold=200):
        self.threshold = threshold

    def detect(self, input_path, output_path):
        
        img = cv2.imread(input_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (320,480))
        image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        image = 255 - image

        # apply threshold
        bw = closing(image > self.threshold, square(3))

        # label image regions
        label_image = label(bw)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img)

        for i, region in enumerate(regionprops(label_image)):
            
            # take regions with large enough areas
            if region.area >= 300:
                #print(f"Object {i}: {region.bbox}")
                # draw rectangle around segmented coins
                minr, minc, maxr, maxc = region.bbox
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                        fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(rect)


        ax.set_axis_off()
        plt.tight_layout()
        fig.savefig(output_path)


if __name__ == '__main__':

    detector = DetectRubbish()
    detector.detect(input_path='images/test2.jpg', 
                        output_path='result_images/res.jpg')