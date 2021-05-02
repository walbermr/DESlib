import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.utils.estimator_checks import check_estimator

from deslib.des.knora_u import KNORAU


def test_check_estimator():
    check_estimator(KNORAU())


# Test the estimate competence method receiving n samples as input
def test_estimate_competence_batch(example_estimate_competence,
                                   create_pool_classifiers):

    X, y, neighbors = example_estimate_competence[0:3]

    query = np.ones((3, 2))
    expected = np.array([[4.0, 3.0, 4.0],
                         [5.0, 2.0, 5.0],
                         [2.0, 5.0, 2.0]])
    knora_u_test = KNORAU(create_pool_classifiers)
    knora_u_test.fit(X, y)

    competences = knora_u_test.estimate_competence(query, neighbors)
    assert np.allclose(competences, expected, atol=0.01)


def test_weights_zero():
    knorau_test = KNORAU()
    competences = np.zeros((1, 3))
    result = knorau_test.select(competences)

    assert np.all(result)


# Test if the class is raising an error when the base classifiers do not
# implements the predict_proba method. In this case the test should not raise
# an error since this class does not require base classifiers that
# can estimate probabilities
def test_predict_proba(create_X_y):
    X, y = create_X_y

    clf1 = Perceptron()
    clf1.fit(X, y)
    KNORAU([clf1, clf1]).fit(X, y)


def test_select():
    knorau_test = KNORAU()
    competences = np.ones(3)
    competences[0] = 0
    expected = np.atleast_2d([False, True, True])
    selected = knorau_test.select(competences)
    assert np.array_equal(expected, selected)


def test_competence_equivalence(example_estimate_competence,
                                   create_pool_classifiers):
    from sklearn.neighbors import DistanceMetric

    def sample_metric(u, v):
        return np.sqrt(np.power(u-v, 2))

    knorau_default = KNORAU(create_pool_classifiers)
    knorau_sample = KNORAU(create_pool_classifiers, neighbor_metric=sample_metric)
    
    X, y, neighbors = example_estimate_competence[0:3]

    query = np.ones((3, 2))
    expected = np.array([[4.0, 3.0, 4.0],
                         [5.0, 2.0, 5.0],
                         [2.0, 5.0, 2.0]])

    knorau_default.fit(X, y)
    knorau_sample.fit(X, y)

    competences_default = knorau_default.estimate_competence(query, neighbors)
    competences_sample = knorau_sample.estimate_competence(query, neighbors)

    assert np.allclose(competences_default, expected)
    assert np.allclose(competences_sample, expected)
    assert np.allclose(competences_default, competences_sample)
    
