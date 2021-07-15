import time
import math
from pathlib import Path
from typing import Union, List, Optional

import typer
import numpy as np
import pandas as pd

import torch
import torch.optim as optim
import torch.multiprocessing
from torch.nn import CrossEntropyLoss

from .utils.config_parser import ConfigParser
from .utils.plotter import plot_losses
from .utils.preprocessor import Preprocessor
from .utils.sampler import StatefulSampler, NeighborSamplerWithWeights
from .utils.common import extend_path, cyan, magenta, Device
from .model.model import Bionic
from .model.loss import masked_scaled_mse


class Trainer:
    def __init__(self, config: Union[Path, dict]):
        """Defines the relevant training and forward pass logic for BIONIC.

        A model is trained by calling `train()` and the resulting gene embeddings are
        obtained by calling `forward()`.

        Args:
            config (Union[Path, dict]): Path to config file or dictionary containing config
                parameters.
        """

        typer.secho("Using CUDA", fg=typer.colors.GREEN) if Device() == "cuda" else typer.secho(
            "Using CPU", fg=typer.colors.RED
        )

        self.params = self._parse_config(
            config
        )  # parse configuration and load into `params` namespace
        self.writer = (
            self._init_tensorboard()
        )  # create `SummaryWriter` for tensorboard visualization
        self.index, self.masks, self.train_mask, self.test_mask, self.weights, self.features, self.encoded_labels, self.label_encoder, self.adj = self._preprocess_inputs()
        self.train_loaders = self._make_train_loaders()
        self.test_loaders = self._make_test_loaders()
        self.inference_loaders = self._make_inference_loaders()
        self.model, self.optimizer = self._init_model()

    def _parse_config(self, config):
        print(config)
        cp = ConfigParser(config)
        return cp.parse()

    def _init_tensorboard(self):
        if self.params.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            return SummaryWriter(flush_secs=10)
        return None

    def _preprocess_inputs(self):
        preprocessor = Preprocessor(
            self.params.names, delimiter=self.params.delimiter, svd_dim=self.params.svd_dim,
        )
        return preprocessor.process()

    def _make_train_loaders(self):
        return [
            NeighborSamplerWithWeights(
                ad,
                sizes=[10] * self.params.gat_shapes["n_layers"],
                batch_size=self.params.batch_size,
                shuffle=False,
                sampler=StatefulSampler(torch.arange(len(self.index))),
            )
            for ad in self.adj
        ]

    def _make_test_loaders(self):
        test_size = len(self.index[self.test_mask])
        return [
            NeighborSamplerWithWeights(
                ad,
                sizes=[10] * self.params.gat_shapes["n_layers"],
                batch_size=test_size,
                shuffle=False,
                sampler=StatefulSampler(torch.arange(len(self.index))),
            )
            for ad in self.adj
        ]

    def _make_inference_loaders(self):
        return [
            NeighborSamplerWithWeights(
                ad,
                sizes=[-1] * self.params.gat_shapes["n_layers"],  # all neighbors
                batch_size=1,
                shuffle=False,
                sampler=StatefulSampler(torch.arange(len(self.index))),
            )
            for ad in self.adj
        ]

    def _init_model(self):
        model = Bionic(
            len(self.index),
            self.params.gat_shapes,
            self.params.embedding_size,
            len(self.encoded_labels.unique()),
            len(self.adj),
            svd_dim=self.params.svd_dim,
        )
        print(f'num of class:{len(self.encoded_labels.unique())}')
        model.apply(self._init_model_weights)

        # Load pretrained model
        # TODO: refactor this
        if self.params.load_pretrained_model:
            typer.echo("Loading pretrained model...")
            model.load_state_dict(torch.load(f"models/{self.params.out_name}_model.pt"))

        # Push model to device
        model.to(Device())

        optimizer = optim.Adam(model.parameters(), lr=self.params.learning_rate, weight_decay=0.0)

        return model, optimizer

    def _init_model_weights(self, model):
        if hasattr(model, "weight"):
            if type(model) == torch.nn.Linear:
                if self.params.initialization == "kaiming":
                    torch.nn.init.kaiming_uniform_(model.weight, a=0.1)
                elif self.params.initialization == "xavier":
                    torch.nn.init.xavier_uniform_(model.weight, gain=1)
                else:
                    raise ValueError(
                        f"The initialization scheme {self.params.initialization} \
                        provided is not supported"
                    )

    def train(self, verbosity: Optional[int] = 1):
        """Trains BIONIC model.

        TODO: this should be refactored

        Args:
            verbosity (int): 0 to supress printing (except for progress bar), 1 for regular printing.
        """

        # Track losses per epoch.
        train_loss = []
        val_acc_lst = []

        best_loss = None
        best_acc = None
        best_state = None

        # Train model.
        for epoch in range(self.params.epochs):

            time_start = time.time()

            # Track average loss across batches.
            epoch_losses = np.zeros(len(self.adj))

            if bool(self.params.sample_size):
                rand_net_idxs = np.random.permutation(len(self.adj))
                idx_split = np.array_split(
                    rand_net_idxs, math.floor(len(self.adj) / self.params.sample_size)
                )
                for rand_idxs in idx_split:
                    train_conf, train_output, train_label, test_conf, test_output, test_label, losses, train_acc, val_acc = self._train_step(rand_idxs)
                    # test_conf, test_output, test_label, val_losses, val_acc = self._test_step(rand_idxs)
                    for idx, loss in zip(rand_idxs, losses):
                        epoch_losses[idx] += loss

            else:
                train_conf, train_output, train_label, test_conf, test_output, test_label, losses, train_acc, val_acc = self._train_step()
                # test_conf, test_output, test_label, val_losses, val_acc = self._test_step()

                epoch_losses = [
                    ep_loss + b_loss.item() / (len(self.index) / self.params.batch_size)
                    for ep_loss, b_loss in zip(epoch_losses, losses)
                ]

            if verbosity:
                progress_string = self._create_progress_string(epoch, epoch_losses, train_acc, val_acc, time_start)
                typer.echo(progress_string)

            # Add loss data to tensorboard visualization
            if self.params.use_tensorboard:
                if len(self.adj) <= 10:
                    writer_dct = {name: loss for name, loss in zip(self.names, epoch_losses)}
                    writer_dct["Total"] = sum(epoch_losses)
                    self.writer.add_scalars("Reconstruction Errors", writer_dct, epoch)

                else:
                    self.writer.add_scalar("Total Reconstruction Error", sum(epoch_losses), epoch)

            train_loss.append(epoch_losses)
            val_acc_lst.append(val_acc)

            # Store best parameter set
            # if not best_loss or sum(epoch_losses) < best_loss:
            if not best_acc or sum(val_acc_lst) > best_acc:
                best_loss = sum(epoch_losses)
                best_acc = sum(val_acc_lst)
                # print(f'val_acc_lst {val_acc_lst}')
                state = {
                    "epoch": epoch + 1,
                    "state_dict": self.model.state_dict(),
                    "best_loss": best_loss,
                    "best_acc": best_acc,
                }
                best_state = state

                train_target = self.label_encoder.inverse_transform(train_label)
                train_pred = self.label_encoder.inverse_transform(train_output)
                test_target = self.label_encoder.inverse_transform(test_label)
                test_pred = self.label_encoder.inverse_transform(test_output)
                train_result_df = pd.DataFrame({'target': train_target,
                                        'prediction': train_pred,
                                        'confidence': train_conf}, index=self.index[self.train_mask])
                test_result_df = pd.DataFrame({'target': test_target,
                                                'prediction': test_pred,
                                                'confidence': test_conf}, index=self.index[self.test_mask])
                train_result_df.to_csv(f'checkpoints/{self.params.out_name}_train_results.csv')
                test_result_df.to_csv(f'checkpoints/{self.params.out_name}_test_results.csv')
                torch.save(state, f'checkpoints/{self.params.out_name}_model.pt')

        if self.params.use_tensorboard:
            self.writer.close()

        self.train_loss, self.best_state = train_loss, best_state
        # print(self.best_state)

    def _train_step(self, rand_net_idx=None):
        """Defines training behaviour.
        """

        # Get random integers for batch.
        rand_int = StatefulSampler.step(len(self.index))
        train_rand_int = rand_int[self.train_mask]
        test_rand_int = rand_int[self.test_mask]

        # print(f'train rand_int is {rand_int}')
        int_splits = torch.split(rand_int, self.params.batch_size)
        train_int_splits = torch.split(train_rand_int, self.params.batch_size)
        test_int_splits = torch.split(test_rand_int, self.params.batch_size)
        batch_features = self.features
        batch_labels = self.encoded_labels[self.train_mask]
        union_train_mask = self.masks[self.train_mask,:]

        # Initialize loaders to current batch.
        if bool(self.params.sample_size):
            batch_train_loaders = [self.train_loaders[i] for i in rand_net_idx]
            if isinstance(self.features, list):
                batch_features = [self.features[i] for i in rand_net_idx]

            # Subset `masks` tensor.
            mask_splits = torch.split(self.masks[:, rand_net_idx][rand_int], self.params.batch_size)
            train_label_splits = torch.split(self.encoded_labels[:, rand_net_idx][train_rand_int], self.params.batch_size)
            test_label_splits = torch.split(self.encoded_labels[:, rand_net_idx][test_rand_int], self.params.batch_size)


        else:
            batch_train_loaders = self.train_loaders
            mask_splits = torch.split(self.masks[rand_int], self.params.batch_size)
            train_label_splits = torch.split(self.encoded_labels[train_rand_int], self.params.batch_size)
            test_label_splits = torch.split(self.encoded_labels[test_rand_int], self.params.batch_size)


            if isinstance(self.features, list):
                batch_features = self.features

        # List of losses.
        losses = [0.0 for _ in range(len(batch_train_loaders))]

        # Get the data flow for each input, stored in a tuple.
        train_max_idx_class_list = []
        train_max_scores_list = []
        test_max_idx_class_list = []
        test_max_scores_list = []
        train_labels_list = []
        test_labels_list = []

        for batch_masks, train_labels, test_labels, node_ids, train_ids, test_ids, *data_flows in zip(mask_splits, train_label_splits, test_label_splits, int_splits, train_int_splits, test_int_splits, *batch_train_loaders):
            self.optimizer.zero_grad()
            cross_entropy_loss = CrossEntropyLoss()
            if bool(self.params.sample_size):
                training_datasets = [self.adj[i] for i in rand_net_idx]
                output_adj, _, _, _, output = self.model(
                    training_datasets,
                    data_flows,
                    batch_features,
                    train_labels,
                    batch_masks,
                    rand_net_idxs=rand_net_idx,
                )
                train_output = output[train_ids, :]
                test_output = output[test_ids, :]
                curr_ce_losses = [
                    cross_entropy_loss(
                          train_output, train_labels.long().to(Device())
                    )
                    for j, i in enumerate(rand_net_idx)
                ]
                curr_mse_losses = [
                    masked_scaled_mse(
                        output_adj, self.adj[i], self.weights[i], node_ids, batch_masks[:, i]
                    )
                    for i in range(len(self.adj))
                ]
            else:
                training_datasets = self.adj
                # print(f'batch_features: {batch_features}')
                # print(f'training_datasets: {training_datasets}')
                # print(f'batch_labels.shape: {batch_labels.shape}')
                # print(f'batch_masks.shape: {batch_masks.shape}')
                # print(f'data_flows: {data_flows}')
                output_adj, _, _, _, output = self.model(
                    training_datasets, data_flows, batch_features, train_labels, batch_masks
                )
                train_output = output[train_ids, :]
                test_output = output[test_ids, :]
                
                # print(f'batch_labels.shape: {batch_labels.shape}')
                curr_ce_losses = [
                    cross_entropy_loss(
                          train_output, train_labels.long().to(Device())
                    )
                    for i in range(len(self.adj))
                ]
                curr_mse_losses = [
                    masked_scaled_mse(
                        output_adj, self.adj[i], self.weights[i], node_ids, batch_masks[:, i]
                    )
                    for i in range(len(self.adj))
                ]

            losses = [loss + curr_ce_loss for loss, curr_ce_loss in zip(losses, curr_ce_losses)]
            # losses = [loss + curr_ce_loss + curr_mse_loss for loss, curr_ce_loss, curr_mse_loss in zip(losses, curr_ce_losses, curr_mse_losses)]

            # loss_sum = sum(curr_ce_losses) + sum(curr_mse_losses)
            print(curr_ce_losses[0])
            print(self.model)
            # print(self.model[0].weight.grad)
            loss_sum = sum(curr_ce_losses) 
            # TODO: 1. grad check
            #       2. index check (random perm vs. true label)
            #       3. standardization
            #       note: should be able to overfit the training labels with JUST cross-entropy loss
            loss_sum.backward()
            softmax = torch.nn.Softmax(dim=1)

            train_max_scores, train_max_idx_class = softmax(train_output).max(dim=1)
            train_max_idx_class_list.extend(list(train_max_idx_class.detach().cpu().numpy()))
            train_max_scores_list.extend(list(train_max_scores.detach().cpu().numpy()))

            test_max_scores, test_max_idx_class = softmax(test_output).max(dim=1)
            test_max_idx_class_list.extend(list(test_max_idx_class.detach().cpu().numpy()))
            test_max_scores_list.extend(list(test_max_scores.detach().cpu().numpy()))

            train_labels_list.extend(list(train_labels.detach().cpu().numpy()))
            test_labels_list.extend(list(test_labels.detach().cpu().numpy()))
            print(f'train_output: {train_labels}')
            print(f'test_output: {train_max_idx_class}')

            self.optimizer.step()
        train_acc = (np.array(train_max_idx_class_list) == np.array(train_labels_list)).astype(np.uint8).sum() / len(train_labels_list)
        test_acc = (np.array(test_max_idx_class_list) == np.array(test_labels_list)).astype(np.uint8).sum() / len(test_labels_list)

        return train_max_scores_list, train_max_idx_class_list, train_labels_list, test_max_scores_list, test_max_idx_class_list, test_labels_list, losses, train_acc, test_acc

    def _test_step(self, rand_net_idx=None):
        """Defines training behaviour.
        """

        # Get random integers for batch.
        test_size = len(self.index[self.test_mask])
        rand_int = StatefulSampler.step(len(self.index))
        test_rand_int = rand_int[self.test_mask]
        # print(f'test rand_int is {rand_int}')

        int_splits = torch.split(rand_int, self.params.batch_size)
        test_int_splits = torch.split(test_rand_int, test_size)
        batch_features = self.features
        batch_labels = self.encoded_labels[self.test_mask]
        union_test_mask = self.masks[self.test_mask,:]

        # Initialize loaders to current batch.
        if bool(self.params.sample_size):
            batch_test_loaders = [self.test_loaders[i] for i in rand_net_idx]
            if isinstance(self.features, list):
                batch_features = [self.features[i] for i in rand_net_idx]

            # Subset `masks` tensor.
            mask_splits = torch.split(self.masks[:, rand_net_idx][rand_int], self.params.batch_size)
            label_splits = torch.split(self.encoded_labels[test_rand_int], self.params.batch_size)

        else:
            batch_test_loaders = self.test_loaders
            mask_splits = torch.split(self.masks[rand_int], self.params.batch_size)
            label_splits = torch.split(self.encoded_labels[test_rand_int], self.params.batch_size)

            if isinstance(self.features, list):
                batch_features = self.features

        # List of losses.
        losses = [0.0 for _ in range(len(batch_test_loaders))]
        max_idx_class_list = []
        max_scores_list = []
        batch_labels_list = []

        # Get the data flow for each input, stored in a tuple.
        for batch_masks, batch_labels, node_ids, test_ids, *data_flows in zip(mask_splits, label_splits, int_splits, test_int_splits, *batch_test_loaders):
            
            self.optimizer.zero_grad()
            cross_entropy_loss = CrossEntropyLoss()
            if bool(self.params.sample_size):
                training_datasets = [self.adj[i] for i in rand_net_idx]
                output_adj, _, _, _, output = self.model(
                    training_datasets,
                    data_flows,
                    batch_features,
                    batch_labels,
                    batch_masks,
                    rand_net_idxs=rand_net_idx,
                )

                test_output = output[test_ids, :]

                curr_ce_losses = [
                    cross_entropy_loss(
                          test_output, batch_labels.long().to(Device())
                    )
                    for j, i in enumerate(rand_net_idx)
                ]
                curr_mse_losses = [
                    masked_scaled_mse(
                        output_adj, self.adj[i], self.weights[i], node_ids, batch_masks[:, j]
                    )
                    for j, i in enumerate(rand_net_idx)
                ]
            else:
                training_datasets = self.adj
                output_adj, _, _, _, output = self.model(
                    training_datasets, data_flows, batch_features, batch_labels, batch_masks
                )
                test_output = output[test_ids, :]

                curr_ce_losses = [
                    cross_entropy_loss(
                          test_output, batch_labels.long().to(Device())
                    )
                    for i in range(len(self.adj))
                ]
                curr_mse_losses = [
                    masked_scaled_mse(
                        output_adj, self.adj[i], self.weights[i], node_ids, batch_masks[:, i]
                    )
                    for i in range(len(self.adj))
                ]

            losses = [loss + curr_ce_loss + curr_mse_loss for loss, curr_ce_loss, curr_mse_loss in zip(losses, curr_ce_losses, curr_mse_losses)]
            loss_sum = sum(curr_ce_losses) + sum(curr_mse_losses)
            loss_sum.backward()
            softmax = torch.nn.Softmax(dim=1)
            max_scores, max_idx_class = softmax(test_output).max(dim=1)
            max_idx_class_list.extend(list(max_idx_class.detach().cpu().numpy()))
            batch_labels_list.extend(list(batch_labels.detach().cpu().numpy()))
            max_scores_list.extend(list(max_scores.detach().cpu().numpy()))
            # print(max_idx_class)
            # print(max_scores)
            # acc = (max_idx_class == batch_labels).sum().item() / batch_labels.size(0)

            self.optimizer.step()
        acc = (np.array(max_idx_class_list) == np.array(batch_labels_list)).astype(np.uint8).sum() / len(batch_labels_list)
        return max_scores_list, max_idx_class_list, batch_labels_list, losses, acc

    def _create_progress_string(
        self, epoch: int, epoch_losses: List[float], acc: float, val_acc: float, time_start: float
    ) -> str:
        """Creates a training progress string to display.
        """
        sep = magenta("|")

        progress_string = (
            f"{cyan('Epoch')}: {epoch + 1} {sep} "
            f"{cyan('Train Acc')}: {acc:.6f} {sep} "
            f"{cyan('Val Acc')}: {val_acc:.6f} {sep} "
            f"{cyan('Loss Total')}: {sum(epoch_losses):.6f} {sep} "
        )
        if len(self.adj) <= 10:
            for i, loss in enumerate(epoch_losses):
                progress_string += f"{cyan(f'Loss {i + 1}')}: {loss:.6f} {sep} "
        progress_string += f"{cyan('Time (s)')}: {time.time() - time_start:.4f}"
        return progress_string

    def forward(self, verbosity: Optional[int] = 1):
        """Runs the forward pass on the trained BIONIC model.

        TODO: this should be refactored

        Args:
            verbosity (int): 0 to supress printing (except for progress bar), 1 for regular printing.
        """
        # Begin inference
        self.model.load_state_dict(
            self.best_state["state_dict"]
        )  # Recover model with lowest reconstruction loss
        if verbosity:
            typer.echo(
                (
                    f"""Loaded best model from epoch {magenta(f"{self.best_state['epoch']}")} """
                    f"""with loss {magenta(f"{self.best_state['best_loss']:.6f}")}"""
                )
            )

        self.model.eval()
        StatefulSampler.step(len(self.index), random=False)
        emb_list = []
        conf_list = []
        pred_list = []

        # Build embedding one node at a time
        # TODO: add verbosity control
        with typer.progressbar(
            zip(self.masks, self.index, *self.inference_loaders),
            label=f"{cyan('Forward Pass')}:",
            length=len(self.index),
        ) as progress:
            for i, (mask, idx, *data_flows) in enumerate(progress):
                mask = mask.reshape((1, -1))
                dot, emb, _, learned_scales, pred  = self.model(
                    self.adj, data_flows, self.features, self.encoded_labels[i], mask, evaluate=True
                )
                softmax = torch.nn.Softmax(dim=1)
                max_scores, max_idx_class = softmax(pred).max(dim=1)
                emb_list.append(emb.detach().cpu().numpy().astype(np.float16))
                conf_list.append(max_scores.detach().cpu().numpy().astype(np.float16)[0])
                pred_list.append(max_idx_class.detach().cpu().numpy().astype(np.uint8)[0])

        emb_features = np.concatenate(emb_list)
        train_emb = emb_features[self.train_mask, :]
        train_emb_df = pd.DataFrame(train_emb, index=self.index[self.train_mask])
        train_emb_df.to_csv(extend_path(self.params.out_name, "_train_features.tsv"), sep="\t")
        test_emb = emb_features[self.test_mask, :]
        test_emb_df = pd.DataFrame(test_emb, index=self.index[self.test_mask])
        test_emb_df.to_csv(extend_path(self.params.out_name, "_test_features.tsv"), sep="\t")

        conf = np.array(conf_list)
        pred = np.array(pred_list)
        target = self.label_encoder.inverse_transform(self.encoded_labels)
        pred = self.label_encoder.inverse_transform(pred)
        train_target = target[self.train_mask]
        test_target = target[self.test_mask]
        train_conf = conf[self.train_mask]
        test_conf = conf[self.test_mask]
        train_pred = pred[self.train_mask]
        test_pred = pred[self.test_mask]
        train_result_df = pd.DataFrame({'target': train_target,
                                        'prediction': train_pred,
                                        'confidence': train_conf}, index=self.index[self.train_mask])
        test_result_df = pd.DataFrame({'target': test_target,
                                        'prediction': test_pred,
                                        'confidence': test_conf}, index=self.index[self.test_mask])
        train_result_df.to_csv(extend_path(self.params.out_name, "_train_results.tsv"), sep="\t")
        test_result_df.to_csv(extend_path(self.params.out_name, "_test_results.tsv"), sep="\t")

        # Free memory (necessary for sequential runs)
        if Device() == "cuda":
            torch.cuda.empty_cache()

        # Create visualization of integrated features using tensorboard projector
        if self.params.use_tensorboard:
            self.writer.add_embedding(emb, metadata=self.index)

        # Output loss plot
        if self.params.plot_loss:
            if verbosity:
                typer.echo("Plotting loss...")
            plot_losses(
                self.train_loss, self.params.names, extend_path(self.params.out_name, "_loss.png")
            )

        # Save model
        if self.params.save_model:
            if verbosity:
                typer.echo("Saving model...")
            torch.save(self.model.state_dict(), extend_path(self.params.out_name, "_model.pt"))

        # Save internal learned network scales
        if self.params.save_network_scales:
            if verbosity:
                typer.echo("Saving network scales...")
            learned_scales = pd.DataFrame(
                learned_scales.detach().cpu().numpy(), columns=self.params.names
            ).T
            learned_scales.to_csv(
                extend_path(self.params.out_name, "_network_weights.tsv"), header=False, sep="\t"
            )

        typer.echo(magenta("Complete!"))
