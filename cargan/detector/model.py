"""Abstract detection model for interference

"""


class ObjectDetector(object):
    def __init__(self, model, label_map):
        """Constructor
        """
        self.model     = model
        self.label_map = label_map

    def predict(self, image):

        boxes, classes, scores = self.model.predict(image)

        return boxes, classes, scores

