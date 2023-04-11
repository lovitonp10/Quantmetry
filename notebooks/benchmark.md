---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.5
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
%matplotlib inline
import logging
import hydra
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from itertools import islice

from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.dataset.repository.datasets import get_dataset

from load.load_prepare import prepare_data
from domain.forecasters import TFTForecaster
# , InformerForecaster
from omegaconf import DictConfig, OmegaConf
from configs import Configs
```

```python
%load_ext autoreload
%autoreload 2
```

```python
with hydra.initialize(version_base="1.3", config_path="configs"):
    cfgHydra = hydra.compose(config_name="config")

cfg = OmegaConf.to_object(cfgHydra)
cfg: Configs = Configs(**cfg)
```

```python
cfg
```

```python
dataset = get_dataset("traffic")
# train_dataset, test_dataset = prepare_data(cfg=cfg)
```

<!-- #region heading_collapsed=true -->
# Informer
<!-- #endregion -->

```python hidden=true
estimator = InformerForecaster(
    freq=dataset.metadata.freq,
    prediction_length=dataset.metadata.prediction_length,
    context_length=dataset.metadata.prediction_length*7,

    #
    num_feat_static_cat=1,
    cardinality=[321],
    embedding_dimension=[3],

    # attention hyper-params
    dim_feedforward=32,
    num_encoder_layers=2,
    num_decoder_layers=2,
    nhead=2,
    activation="relu",

)
```

```python hidden=true
predictor_informer = estimator.train(
    training_data = dataset.train,
    cache_data=True,
    #num_workers=8,
    #shuffle_buffer_length=1024
)
```

```python hidden=true
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.dataset.repository.datasets import get_dataset

from domain.estimator_informer import InformerEstimator
```

```python hidden=true
estimator = InformerEstimator(
    freq=dataset.metadata.freq,
    prediction_length=dataset.metadata.prediction_length,
    context_length=dataset.metadata.prediction_length*7,

    #
    num_feat_static_cat=1,
    cardinality=[321],
    embedding_dimension=[3],

    # attention hyper-params
    dim_feedforward=32,
    num_encoder_layers=2,
    num_decoder_layers=2,
    nhead=2,
    activation="relu",

)
```

```python hidden=true
predictor_informer = estimator.train(
    training_data = dataset.train,
    cache_data=True,
    #num_workers=8,
    #shuffle_buffer_length=1024
)
```

# TFT

```python
estimator = TFTForecaster(
        cfg_model = cfg.model,
        cfg_train= cfg.train,
    cfg_dataset=cfg.dataset,
)
```

```python
predictor = estimator.train(
    training_data = dataset.train,
)
```

```python
predictor.
```

```python
forecast_it, ts_it = make_evaluation_predictions(
    dataset=dataset.test,
    predictor=predictor
)
```

```python
forecasts = list(forecast_it)
```

```python
forecasts
```

```python
tss = list(ts_it)
```

```python
evaluator = Evaluator(
    quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    seasonality=1,
    )
```

```python
agg_metrics, ts_metrics = evaluator(
    ts_iterator = tss,
    fcst_iterator = forecasts,
    )
agg_metrics
```

```python
plt.figure(figsize=(20, 15))
date_formater = mdates.DateFormatter('%b, %d')
plt.rcParams.update({'font.size': 15})

for idx, (forecast, ts) in islice(enumerate(zip(forecasts, tss)), 9):
    ax = plt.subplot(3, 3, idx+1)

    plt.plot(ts[-4 * dataset.metadata.prediction_length:], label="target", )
    forecast.plot( color='g')
    plt.xticks(rotation=60)
    plt.title(forecast.item_id)
    ax.xaxis.set_major_formatter(date_formater)

plt.gcf().tight_layout()
plt.legend()
plt.show()
```
