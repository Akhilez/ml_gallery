# Numpy is needed because OpenCV images in python are actually numpy arrays.
import cv2
import numpy


class iris_detection():
    def __init__(self, image_path):
        '''
        initialize the class and set the class attributes
        '''
        self._img = None
        self._img_path = image_path
        self._pupil = None

    def load_image(self):
        '''
        load the image based on the path passed to the class
        it should use the method cv2.imread to load the image
        it should also detect if the file exists
        '''
        self._img = cv2.imread(self._img_path)
        # If the image doesn't exists or is not valid then imread returns None
        if self._img is None:
            return False
        else:
            return True

    def convert_to_gray_scale(self):
        self._img = cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)

    def detect_pupil(self):
        '''
        This method should use cv2.findContours and cv2.HoughCircles() function from cv2 library to find the pupil
        and then set the coordinates for pupil circle coordinates
        '''
        # First binarize the image so that findContours can work correctly.
        _, thresh = cv2.threshold(self._img, 100, 255, cv2.THRESH_BINARY)
        # Now find the contours and then find the pupil in the contours.
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        c = cv2.HoughCircles(contours, cv2.HOUGH_GRADIENT, 2, self._img.shape[0] / 2)
        # Then mask the pupil from the image and store it's coordinates.
        for l in c:
            # OpenCV returns the circles as a list of lists of circles
            for circle in l:
                center = (circle[0], circle[1])
                radius = circle[2]
                cv2.circle(self._img, center, radius, (0, 0, 0), thickness=-1)
                self._pupil = (center[0], center[1], radius)

    def detect_iris(self):
        '''
        This method should use the background subtraction technique to isolate the iris from the original image
        It should use the coordinates from the detect_pupil to get a larger circle using cv2.HoughCircles()
        '''
        _, t = cv2.threshold(self._img, 195, 255, cv2.THRESH_BINARY)
        contours, _, _ = cv2.findContours(t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Find the iris using the radius of the pupil as input.
        c = cv2.HoughCircles(contours, cv2.HOUGH_GRADIENT, 2, self._pupil[2] * 2, param2=150)

        for l in c:
            for circle in l:
                center = (self._pupil[0], self._pupil[1])
                radius = circle[2]
                # This creates a black image and draws an iris-sized white circle in it.
                mask = numpy.zeros((self._img.shape[0], self._img.shape[1], 1), numpy.uint8)
                cv2.circle(mask, center, radius, (255, 255, 255), thickness=-1)
                # Mask the iris and crop everything outside of its radius.
                self._img = cv2.bitwise_and(self._img, mask)

    def start_detection(self):
        '''
        This is the main method that will be called to detect the iris
        it will call all the previous methods in the following order:
        load_image
        convert_to_gray_scale
        detect_pupil
        detect_iris
        then it should display the resulting image with the iris only
        using the method cv2.imshow
        '''
        if (self.load_image()):
            self.convert_to_gray_scale()
            self.detect_pupil()
            self.detect_iris()
            cv2.imshow("Result", self._img)
            cv2.waitKey(0)
        else:
            print('Image file "' + self._img_path + '" could not be loaded.')


if __name__ == '__main__':
    # id = iris_detection('c:\\temp\\eye_image.jpg')
    id = iris_detection('/Users/akhil/code/ml_gallery/ml_py/data/pupil/test/images/C1_S1_I1.tiff')
    id.start_detection()
