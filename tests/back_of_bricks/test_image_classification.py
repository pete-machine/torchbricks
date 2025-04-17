import torch

from torchbricks.bag_of_bricks.image_classification import ImageClassifier


def test_image_classifier_average_pooling():
    batch_size = 2
    input_features = 5
    n_classes = 10
    HW = 20
    image_classifier = ImageClassifier(num_classes=n_classes, n_features=input_features, use_average_pooling=True)
    embedding = torch.zeros((batch_size, input_features, HW, HW))
    logits, probabilities, class_prediction = image_classifier(embedding)

    assert logits.shape == (batch_size, n_classes)
    assert probabilities.shape == (batch_size, n_classes)
    assert class_prediction.shape == (batch_size,)


def test_image_classifier_no_average_pooling():
    batch_size = 2
    input_features = 5
    n_classes = 10
    image_classifier = ImageClassifier(num_classes=n_classes, n_features=input_features, use_average_pooling=False)
    embedding = torch.zeros((batch_size, input_features))
    logits, probabilities, class_prediction = image_classifier(embedding)

    assert logits.shape == (batch_size, n_classes)
    assert probabilities.shape == (batch_size, n_classes)
    assert class_prediction.shape == (batch_size,)
