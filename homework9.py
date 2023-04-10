############################################################
# CIS 521: Homework 8
############################################################

student_name = "Jingjing Bai"

############################################################
# Imports
############################################################

import homework9_data as data

# Include your imports here, if any are used.



############################################################
# Section 1: Perceptrons
############################################################

class BinaryPerceptron(object):

    def __init__(self, examples, iterations):
        example, _ = examples[0]
        self.w_map = { key: 0 for key in example.keys()}

        for __ in range(iterations):
            for x, y in examples:
                predict_label = self.predict(x)

                if predict_label == y:
                    continue
                # update weight map if mislabel.
                self.w_map = {k : v + x.get(k, 0) * (1 if y else - 1) for k, v in self.w_map.items()}

    def predict(self, x):
        return sum([v * self.w_map.get(key, 0) for key, v in x.items()]) > 0

class MulticlassPerceptron(object):

    def __init__(self, examples, iterations):
        pass
    
    def predict(self, x):
        pass

############################################################
# Section 2: Applications
############################################################

class IrisClassifier(object):

    def __init__(self, data):
        pass

    def classify(self, instance):
        pass

class DigitClassifier(object):

    def __init__(self, data):
        pass

    def classify(self, instance):
        pass

class BiasClassifier(object):

    def __init__(self, data):
        pass

    def classify(self, instance):
        pass

class MysteryClassifier1(object):

    def __init__(self, data):
        pass

    def classify(self, instance):
        pass

class MysteryClassifier2(object):

    def __init__(self, data):
        pass

    def classify(self, instance):
        pass

############################################################
# Section 3: Feedback
############################################################

feedback_question_1 = 0

feedback_question_2 = """
Type your response here.
Your response may span multiple lines.
Do not include these instructions in your response.
"""

feedback_question_3 = """
Type your response here.
Your response may span multiple lines.
Do not include these instructions in your response.
"""
