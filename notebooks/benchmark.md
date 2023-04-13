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
cd ..
```

```python
import sys, os
sys.path.append(os.path.join(os.getcwd(),"dl4tsf/"))
```

```python
%matplotlib inline
import hydra
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from gluonts.evaluation import make_evaluation_predictions

from load.dataloaders import CustomDataLoader
from domain.forecasters import TFTForecaster

# , InformerForecaster
from omegaconf import OmegaConf
from configs import Configs
```

```python
pwd

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
cfg.dataset
```

# TFT

```python
cfg.model
```

```python
loader_data = CustomDataLoader(
    cfg_dataset=cfg.dataset,
    target=cfg.dataset.load['target'],
    cfg_model=cfg.model,
    test_length=5,
)
data_gluonts = loader_data.get_gluonts_format()
```

```python
estimator = TFTForecaster(
        cfg_model = cfg.model,
        cfg_train= cfg.train,
    cfg_dataset=cfg.dataset,
)
```

```python
estimator.train(
    input_data = data_gluonts.train,
)
```

```python
true_ts, forecasts = estimator.predict(data_gluonts.test)
```

```python
metrics = estimator.evaluate(data_gluonts.test)
metrics
```

## Plot

```python
plt.figure(figsize=(20, 15))
date_formater = mdates.DateFormatter('%b, %d')
plt.rcParams.update({'font.size': 15})

for idx, (forecast, ts) in enumerate(zip(forecasts, true_ts)):
    ts_plot = ts.copy()
    ts_plot.index = ts_plot.index.to_timestamp()
    ax = plt.subplot(3, 3, idx+1)
    plt.plot(ts_plot[-4 * cfg.model.model_config.prediction_length:][0], label="target", )
    forecast.median(axis=1).plot( color='g', label='forecast')
    q_05 = forecast.apply(lambda row: np.quantile(row, 0.025), axis=1)# for 0.5 confidence level
    q_95 = forecast.apply(lambda row: np.quantile(row, 0.975), axis=1)# for 0.5 confidence level
    q_10 = forecast.apply(lambda row: np.quantile(row, 0.05), axis=1)# for 0.9 confidence level
    q_90 = forecast.apply(lambda row: np.quantile(row, 0.95), axis=1)# for 0.9 confidence level
    plt.fill_between(
                ts_plot[-cfg.model.model_config.prediction_length:].index,
                q_05,
                q_95,
                facecolor='blue',
                alpha=0.6,
                interpolate=True)
    plt.fill_between(
                ts_plot[-cfg.model.model_config.prediction_length:].index,
                q_10,
                q_90,
                facecolor='blue',
                alpha=0.4,
                interpolate=True)

    plt.xticks(rotation=60)
#     plt.title(forecast.item_id)
    ax.xaxis.set_major_formatter(date_formater)

plt.gcf().tight_layout()
plt.legend()
plt.show()
```

<!-- #region heading_collapsed=true -->
# Informer (not ready to ignore)
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
