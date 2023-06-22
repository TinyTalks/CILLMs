import datasets
import numpy as np
from accelerate import Accelerator
from omegaconf import open_dict
import hydra
import torch
import time
from datasets.iterable_dataset import IterableDataset
from torch.utils.data import DataLoader

from nanoT5.nanoT5.utils import process_dataset, get_data_collator
from nanoT5.nanoT5.utils import (
    setup_basics,
    train,
    predict,
    eval,
    get_lr_scheduler,
    get_optimizer,
    get_tokenizer,
    get_model,
    get_dataloaders,
    get_config,
)


def load_dataset_splits(args):
    if args.mode == 'pt':

        dataset = datasets.load_dataset('text',
                                        data_files={'train': 'train.txt', 'validation': 'test.txt', 'test': 'test.txt'})

        # dataset = datasets.load_dataset(
        #     'c4',
        #     'en',
        #     streaming=True,
        # )
        #
        # dataset = dataset.remove_columns(
        #     ['timestamp', 'url']
        # )

        dataset_splits = {
            'train': dataset['train'],
            'test': dataset['validation'],
        }

        assert (
                dataset['train'].n_shards == 1024
        ), "We want to have many shards for efficient processing with num_workes in PyTorch dataloader"
    elif args.mode == 'ft':
        dataset_splits = datasets.load_dataset(
            args.data.exec_file_path,
            data_dir=args.data.data_dir,
            task_dir=args.data.task_dir,
            max_num_instances_per_task=args.data.max_num_instances_per_task,
            max_num_instances_per_eval_task=args.data.max_num_instances_per_task
        )
    else:
        raise NotImplementedError

    return dataset_splits


def get_dataloaders(tokenizer, config, args):
    dataset_splits = load_dataset_splits(args)
    dataset = process_dataset(dataset_splits=dataset_splits, args=args, tokenizer=tokenizer)
    data_collator = get_data_collator(tokenizer=tokenizer, config=config,
                                      args=args)

    is_iterable = isinstance(dataset['train'], IterableDataset)

    dataloaders = {}

    for split in ['train', 'test']:
        batch_size = args.optim.batch_size // args.optim.grad_acc

        if split in ['test']:
            batch_size *= 2

        shuffle = (split == 'train') and not is_iterable

        if args.mode == 'ft' and split == 'train':
            assert shuffle is True
        else:
            assert shuffle is False

        dataloaders[split] = DataLoader(
            dataset[split],
            shuffle=shuffle,
            collate_fn=data_collator,
            batch_size=batch_size,
            num_workers=args.data.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    # Add & Check args about data loaders
    with open_dict(args):
        if not is_iterable:
            args.data.train_batches = len(dataloaders['train'])
            args.data.test_batches = len(dataloaders['test'])

        if args.optim.epochs > 0:
            assert not is_iterable
            args.optim.total_steps = (len(dataloaders['train']) // args.optim.grad_acc) * args.optim.epochs

            # We increase eval BS by 2, so decrease number of eval steps
        args.eval.corrected_steps = args.eval.steps / 2

    return dataloaders['train'], dataloaders['test']


@hydra.main(config_path="configs", config_name="default", version_base='1.1')
def main(args):
    accelerator = Accelerator(cpu=args.device == "cpu")
    logger = setup_basics(accelerator, args)
    config = get_config(args)
    model = get_model(args, config)
    tokenizer = get_tokenizer(args)
    optimizer = get_optimizer(model, args)
    lr_scheduler = get_lr_scheduler(optimizer, args, logger)
    train_dataloader, test_dataloader = get_dataloaders(tokenizer, config, args)

    logger.log_args(args)

    (
        model,
        optimizer,
        lr_scheduler,
        train_dataloader,
        test_dataloader,
    ) = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader, test_dataloader
    )

    if args.model.compile:
        model = torch.compile(model)

    with open_dict(args):
        args.current_train_step = 1
        args.current_epoch = 1
        args.last_log = time.time()

    if args.eval_only:
        model.eval()
        with torch.no_grad():
            eval(model, test_dataloader, logger, args, tokenizer)
    elif args.predict_only:
        model.eval()
        with torch.no_grad():
            predict(model, test_dataloader, logger,
                    args, tokenizer)
    else:
        train(model, train_dataloader, test_dataloader, accelerator,
              lr_scheduler, optimizer, logger, args, tokenizer)

    logger.finish()


if __name__ == "__main__":
    main()
