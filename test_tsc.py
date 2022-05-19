"""Tests for pytorch/rnn.py.
Based on: https://github.com/skorch-dev/skorch/blob/master/skorch/tests/test_regressor.py
"""
import numpy as np
import pytest
import torch

class TestTSC:
    @pytest.fixture
    def net(self):
        from src import ConvNet
        return ConvNet(n_in=1000, n_classes=20, max_epochs=10, lr=0.01, 
            batch_size=12, optimizer=torch.optim.SGD, iterator_train__shuffle=False )
    
    @pytest.fixture
    def net_partial_fit(self, net, classification_data):
        X, y = classification_data
        for i in range(1,3):
            net.partial_fit(X, y)
        return net
    
    def test_net_learns(self, net, classification_data):
        X, y = classification_data
        for i in range(0,3):
            net.partial_fit(X, y)
        train_losses = net.history[:, 'train_loss']
        assert train_losses[0] > train_losses[-1]

    def test_predict_predict_proba(self, net_partial_fit, classification_data):
        X = classification_data[0]
        y_pred = net_partial_fit.predict(X)

        # predictions should not be all zeros
        assert not np.allclose(y_pred, 0)
        y_proba = net_partial_fit.predict_proba(X)

        # predict and predict_proba should be identical for regression
        assert np.allclose(y_pred, y_proba, atol=1e-6)

    def test_score(self, net_partial_fit, classification_data):
        X, y = classification_data
        r2_score = net_partial_fit.score(X, y)
        assert r2_score <= 1. and r2_score > 0.9
