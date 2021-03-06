from abc import ABCMeta

import numpy as np
from deslib.base import BaseDS
from deslib.util.aggregation import (weighted_majority_voting_rule,
                                     majority_voting_rule,
                                     aggregate_proba_ensemble_weighted)


class BaseDES(BaseDS):
    """Base class for a Dynamic Ensemble Selection (DES).

    All dynamic ensemble selection techniques should inherit from this class.

    Warning: This class should not be instantiated directly, use
    derived classes instead.

    """
    __metaclass__ = ABCMeta

    def __init__(self, pool_classifiers=None, k=7, DFP=False, with_IH=False,
                 safe_k=None, IH_rate=0.30, mode='selection',
                 needs_proba=False, random_state=None,
                 knn_classifier='knn', knne=False, DSEL_perc=0.5, n_jobs=-1):

        super(BaseDES, self).__init__(pool_classifiers=pool_classifiers,
                                      k=k,
                                      DFP=DFP,
                                      with_IH=with_IH,
                                      safe_k=safe_k,
                                      IH_rate=IH_rate,
                                      needs_proba=needs_proba,
                                      random_state=random_state,
                                      knn_classifier=knn_classifier,
                                      knne=knne,
                                      DSEL_perc=DSEL_perc, n_jobs=n_jobs)
        self.mode = mode

    def estimate_competence(self, query, neighbors, distances=None,
                            predictions=None):
        """Estimate the competence of each base classifier :math:`c_{i}`
        the classification of the query sample x.
        Returns an array containing the level of competence estimated
        for each base classifier. The size of the vector is equals to
        the size of the generated_pool of classifiers.

        Parameters
        ----------
        query : array of shape (n_samples, n_features)
                The test examples

        neighbors : array of shape (n_samples, n_neighbors)
            Indices of the k nearest neighbors according for each test sample

        distances : array of shape (n_samples, n_neighbors)
            Distances of the k nearest neighbors according for each test sample

        predictions : array of shape (n_samples, n_classifiers)
            Predictions of the base classifiers for the test examples.

        Returns
        -------
        competences : array of shape (n_samples, n_classifiers)
            Competence level estimated for each base classifier and test
            example.
        """
        pass

    def estimate_competence_from_proba(self, query, neighbors, probabilities,
                                       distances=None):
        """ estimate the competence of each base classifier :math:`c_{i}`
        the classification of the query sample x, for methods that require
        probabilities.

        Returns an array containing the level of competence estimated
        for each base classifier. The size of the vector is equals to
        the size of the generated_pool of classifiers.

        Parameters
        ----------
        query : array of shape (n_samples, n_features)
                The query sample.

        neighbors : array of shape (n_samples, n_neighbors)
            Indices of the k nearest neighbors according for each test sample.

        distances : array of shape (n_samples, n_neighbors)
            Distances of the k nearest neighbors according for each test
            sample.

        probabilities : array of shape (n_samples, n_classifiers, n_classes)
            Probabilities estimates of each base classifier for all samples.

        Returns
        -------
        competences : array = [n_samples, n_classifiers]
            Competence level estimated for each base classifier and test
            example.
        """
        pass

    def select(self, competences):
        """Select the most competent classifiers to compose an ensemble for
        the classification of the query sample X.

        Parameters
        ----------
        competences : array of shape (n_samples, n_classifiers)
            Estimated competence level of each base classifier for each test
            example.

        Returns
        -------
        selected_classifiers : array of shape (n_samples, n_classifiers)
            Boolean matrix containing True if the base classifier is selected.
            False otherwise.
        """
        pass

    def classify_with_ds(self, query, predictions, probabilities=None,
                         neighbors=None, distances=None, DFP_mask=None):
        """Predicts the label of the corresponding query sample.

        If self.mode == "selection", the selected ensemble is combined using
        the majority voting rule

        If self.mode == "weighting", all base classifiers are used for
        classification, however their influence in the final decision are
        weighted according to their estimated competence level. The weighted
        majority voting scheme is used to combine the decisions of the
        base classifiers.

        If self.mode == "hybrid",  A hybrid Dynamic selection and weighting
        approach is used. First an ensemble with the competent base classifiers
        are selected. Then, their decisions are aggregated using the weighted
        majority voting rule according to its competence level estimates.

        Parameters
        ----------
        query : array of shape (n_samples, n_features)
                The test examples.

        predictions : array of shape (n_samples, n_classifiers)
                      Predictions of the base classifier for all test examples.

        probabilities : array of shape (n_samples, n_classifiers, n_classes)
            Probabilities estimates of each base classifier for all test
            examples. (For methods that always require probabilities from
            the base classifiers).

        neighbors : array of shape (n_samples, n_neighbors)
            Indices of the k nearest neighbors according for each test sample.

        distances : array of shape (n_samples, n_neighbors)
            Distances of the k nearest neighbors according for each test
            sample.

        DFP_mask : array of shape (n_samples, n_classifiers)
            Mask containing 1 for the selected base classifier and 0 otherwise.

        Returns
        -------
        predicted_label : array of shape (n_samples)
                          Predicted class label for each test example.
        """
        if query.ndim < 2:
            query = query.reshape(1, -1)

        if predictions.ndim < 2:
            predictions = predictions.reshape(1, -1)

        if query.shape[0] != predictions.shape[0]:
            raise ValueError(
                'The arrays query and predictions must have the same number'
                ' of samples. query.shape is {}'
                'and predictions.shape is {}'.format(query.shape,
                                                     predictions.shape))

        if self.needs_proba:
            competences = self.estimate_competence_from_proba(
                query,
                neighbors=neighbors,
                distances=distances,
                probabilities=probabilities)
        else:
            competences = self.estimate_competence(query,
                                                   neighbors=neighbors,
                                                   distances=distances,
                                                   predictions=predictions)

        if self.DFP:
            competences = competences * DFP_mask

        if self.mode == "selection":
            # The selected_classifiers matrix is used as a mask to remove
            # the predictions of certain base classifiers.
            selected_classifiers = self.select(competences)
            votes = np.ma.MaskedArray(predictions, ~selected_classifiers)
            predicted_label = majority_voting_rule(votes)

        elif self.mode == "weighting":
            votes = np.atleast_2d(predictions)
            predicted_label = weighted_majority_voting_rule(votes, competences,
                                                            np.arange(
                                                                self.n_classes_
                                                                      ))
        else:
            selected_classifiers = self.select(competences)
            votes = np.ma.MaskedArray(predictions, ~selected_classifiers)
            predicted_label = weighted_majority_voting_rule(votes, competences,
                                                            np.arange(
                                                                self.n_classes_
                                                                      ))

        return predicted_label

    def predict_proba_with_ds(self, query, predictions, probabilities,
                              neighbors=None, distances=None, DFP_mask=None):
        """Predicts the posterior probabilities of the corresponding query.

        If self.mode == "selection", the selected ensemble is used to estimate
        the probabilities. The average rule is used
        to give probabilities estimates.

        If self.mode == "weighting", all base classifiers are used for
        estimating the probabilities, however their influence in the final
        decision are weighted according to their estimated competence level.
        A weighted average method is used to give the probabilities estimates.

        If self.mode == "Hybrid",  A hybrid Dynamic selection and weighting
        approach is used. First an ensemble with the competent base classifiers
        are selected. Then, their decisions are aggregated using a weighted
        average rule to give the probabilities estimates.

        Parameters
        ----------
        query : array of shape (n_samples, n_features)
                The test examples.

        predictions : array of shape (n_samples, n_classifiers)
            Predictions of the base classifier for all test examples.

        probabilities : array of shape (n_samples, n_classifiers, n_classes)
            Probabilities estimates of each base classifier for all samples.

        neighbors : array of shape (n_samples, n_neighbors)
            Indices of the k nearest neighbors according for each test sample

        distances : array of shape (n_samples, n_neighbors)
            Distances of the k nearest neighbors according for each test sample

        DFP_mask : array of shape (n_samples, n_classifiers)
            Mask containing 1 for the selected base classifier and 0 otherwise.

        Returns
        -------
        predicted_proba : array = [n_samples, n_classes]
                          The probability estimates for all test examples.
        """

        if query.shape[0] != probabilities.shape[0]:
            raise ValueError(
                'The arrays query and predictions must have the same number'
                ' of samples. query.shape is {}'
                'and predictions.shape is {}'.format(query.shape,
                                                     predictions.shape))

        if self.needs_proba:
            competences = self.estimate_competence_from_proba(
                query,
                neighbors=neighbors,
                distances=distances,
                probabilities=probabilities)
        else:
            competences = self.estimate_competence(query,
                                                   neighbors=neighbors,
                                                   distances=distances,
                                                   predictions=predictions)

        if self.DFP:
            competences = competences * DFP_mask

        if self.mode == "selection":
            selected_classifiers = self.select(competences)

            # Broadcast the selected classifiers mask
            # to cover the last axis (n_classes):
            selected_classifiers = np.expand_dims(selected_classifiers, axis=2)
            selected_classifiers = np.broadcast_to(selected_classifiers,
                                                   probabilities.shape)
            masked_proba = np.ma.MaskedArray(probabilities,
                                             ~selected_classifiers)

            predicted_proba = np.mean(masked_proba, axis=1)

        elif self.mode == "weighting":
            predicted_proba = aggregate_proba_ensemble_weighted(probabilities,
                                                                competences)
        else:
            selected_classifiers = self.select(competences)

            # Broadcast the selected classifiers mask
            # to cover the last axis (n_classes):
            selected_classifiers = np.expand_dims(selected_classifiers, axis=2)
            selected_classifiers = np.broadcast_to(selected_classifiers,
                                                   probabilities.shape)
            masked_proba = np.ma.MaskedArray(probabilities,
                                             ~selected_classifiers)

            predicted_proba = aggregate_proba_ensemble_weighted(masked_proba,
                                                                competences)

        return predicted_proba

    def _validate_parameters(self):

        super(BaseDES, self)._validate_parameters()

        if not isinstance(self.mode, str):
            raise TypeError(
                'Parameter "mode" should be a string.'
                ' Currently "mode" = {}' .format(type(self.mode)))

        if self.mode not in ['selection', 'hybrid', 'weighting']:
            raise ValueError(
                'Invalid value for parameter "mode".'
                ' "mode" should be one of these options '
                '{selection, hybrid, weighting}')
