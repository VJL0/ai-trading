"""Local quant model suite: leakage-free walk-forward training, net-Sharpe model selection, registry, predictions.

Public API (import from the submodules):

* ``services.research.models.walkforward.train_suite`` -> ``TrainResult``
* ``services.research.models.registry.save_registry`` / ``load_registry`` / ``ModelRegistry``
* ``services.research.models.predict.PredictionService``
* ``services.research.models.config.SuiteConfig``

The quant suite never depends on any AI runtime. NO EDGE IS CLAIMED.
"""
