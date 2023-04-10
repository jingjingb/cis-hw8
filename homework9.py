import homework9_data as data
############################################################
# CIS 521: Homework 8
############################################################

student_name = "Jingjing Bai"

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
                for k in x.keys():
                    if k in self.w_map:
                        self.w_map[k] += x.get(k, 0) * (1 if y else - 1)
                    else:
                        self.w_map[k] = x.get(k, 0) * (1 if y else - 1)

    def predict(self, x):
        return sum([v * self.w_map.get(key, 0) for key, v in x.items()]) > 0

class MulticlassPerceptron(object):

    def __init__(self, examples, iterations):
        # label to weight
        self.l_to_w_map = {y: {} for (x, y) in examples}

        for __ in range(iterations):
            for x, correct_y in examples:
                predicted_label = None
                current_max = -1
                for l in self.l_to_w_map.keys():
                    l_score = self.predict_l_score(x, l)
                    if l_score > current_max:
                        current_max = l_score
                        predicted_label = l
                if predicted_label != correct_y:
                    correct_label_map = self.l_to_w_map.get(correct_y)
                    for key in x.keys():
                        if key in correct_label_map:
                            correct_label_map[key] += x[key]
                        else:
                            correct_label_map[key] = x[key]
                    predicted_label_map = self.l_to_w_map.get(predicted_label)
                    for key in x.keys():
                        if key in predicted_label_map:
                            predicted_label_map[key] -= x[key]
                        else:
                            predicted_label_map[key] = -x[key]

    def predict_l_score(self, x, l):
        l_w_map = self.l_to_w_map.get(l)
        return sum([x[key] * l_w_map.get(key, 0) for key in x.keys()])
    
    def predict(self, x):
        predicted_label = None
        current_max = -1
        for l in self.l_to_w_map.keys():
            l_score = self.predict_l_score(x, l)
            if l_score > current_max:
                current_max = l_score
                predicted_label = l
        return predicted_label

############################################################
# Section 2: Applications
############################################################
def read_data(data):
    return [({i: v for i, v in enumerate(x, 1)}, y) for x, y in data]

def format_input(instance):
    return {i+1: instance[i] for i, x in enumerate(instance, 0)}

class IrisClassifier(object):

    def __init__(self, data):
        self.classifier = MulticlassPerceptron(read_data(data), 100)

    def classify(self, instance):
        return self.classifier.predict(format_input(instance))

class DigitClassifier(object):

    def __init__(self, data):
        self.classifier = MulticlassPerceptron(read_data(data), 10)

    def classify(self, instance):
        return self.classifier.predict(format_input(instance))


class BiasClassifier(object):

    def __init__(self, data):
        self.classifier = BinaryPerceptron([({1: x, 2: 1}, y) for x, y in data], 10)

    def classify(self, instance):
        return self.classifier.predict({1: instance, 2: 1})


class MysteryClassifier1(object):

    def __init__(self, data):
        self.classifier = BinaryPerceptron([({1: x[0]**2 + x[1]**2, 2: 1}, y) for x, y in data], 10)

    def classify(self, instance):
        return self.classifier.predict({1: instance[0]**2 + instance[1]**2, 2: 1})

class MysteryClassifier2(object):

    def __init__(self, data):
        self.classifier = BinaryPerceptron([({1: x[0] * x[1] * x[2]}, y) for x, y in data], 10)

    def classify(self, instance):
        return self.classifier.predict({1: instance[0] * instance[1] * instance[2]})

############################################################
# Section 3: Feedback
############################################################

feedback_question_1 = 5

feedback_question_2 = """
the details of implementation, corner cases
"""

feedback_question_3 = """
application examples are great
"""
