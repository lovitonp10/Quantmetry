import logging

import numpy as np
from accelerate import Accelerator
from configs import Configs
from torch.optim import AdamW


def train(config_model, model, cfg: Configs, train_dataloader):
    accelerator = Accelerator()
    device = accelerator.device
    model.to(device)
    optimizer = AdamW(model.parameters(), **cfg.model.optimizer_config)

    model, optimizer, train_dataloader = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
    )

    logging.info("Training")
    loss_history = []
    model.train()
    for epoch in range(cfg.train.epochs):
        for idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs = model(
                static_categorical_features=batch["static_categorical_features"].to(device)
                if config_model.num_static_categorical_features > 0
                else None,
                static_real_features=batch["static_real_features"].to(device)
                if config_model.num_static_real_features > 0
                else None,
                past_time_features=batch["past_time_features"].to(device),
                past_values=batch["past_values"].to(device),
                future_time_features=batch["future_time_features"].to(device),
                future_values=batch["future_values"].to(device),
                past_observed_mask=batch["past_observed_mask"].to(device),
                future_observed_mask=batch["future_observed_mask"].to(device),
            )
            loss = outputs.loss

            # Backpropagation
            accelerator.backward(loss)
            optimizer.step()

            loss_history.append(loss.item())
            if idx % 100 == 0:
                print(loss.item())

    return model, loss_history


def inference(model, model_config, test_dataloader):

    accelerator = Accelerator()
    device = accelerator.device

    model.to(device)
    model.eval()
    forecasts_ = []

    for batch in test_dataloader:
        outputs = model.generate(
            static_categorical_features=batch["static_categorical_features"].to(device)
            if model_config.num_static_categorical_features > 0
            else None,
            static_real_features=batch["static_real_features"].to(device)
            if model_config.num_static_real_features > 0
            else None,
            past_time_features=batch["past_time_features"].to(device),
            past_values=batch["past_values"].to(device),
            future_time_features=batch["future_time_features"].to(device),
            past_observed_mask=batch["past_observed_mask"].to(device),
        )
        forecasts_.append(outputs.sequences.cpu().numpy())

    forecasts = np.vstack(forecasts_)
    return forecasts
