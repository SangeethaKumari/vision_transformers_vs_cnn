
#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2024.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#   Use is limited to the duration and purpose of the training at SupportVectors.
#
#   Author: SupportVectors AI Training Team
#  -------------------------------------------------------------------------------------------------
#

import joblib

# svlearn
from svlearn_vit_cnn import config
from svlearn.common.utils import ensure_directory
from svlearn_vit_cnn.utils.train_utils import (
    print_trainable_parameters,
    compute_metrics,
    collate_fn,
    make_train_transform,
    make_test_transform,
    prepare_datasets,
    get_device
)

# huggingface
from transformers import (
    ViTImageProcessor,
    AutoImageProcessor,
    ViTForImageClassification,
    ResNetForImageClassification,
    Trainer,
    TrainingArguments
)

#  -------------------------------------------------------------------------------------------------

# configurations
current_task = config['current_task']

data_dir = config['tree-dataset']['path']
raw_dir = data_dir + '/trees'

# Determine task-specific settings based on current_task
if current_task == 'vit_classification':
    processed_dir = data_dir + "/preprocessed"
    results_dir = config['vision-transformer']['results']
    model_name_or_path = config['vision-transformer']['model_name']
    processor = ViTImageProcessor.from_pretrained(model_name_or_path)
    model_class = ViTForImageClassification
elif current_task == 'resnet_classification':
    processed_dir = data_dir + "/preprocessed_resnet"
    results_dir = config['cnn']['results']
    model_name_or_path = config['cnn']['model_name']
    processor = AutoImageProcessor.from_pretrained(model_name_or_path)
    model_class = ResNetForImageClassification
else:
    raise ValueError(f"Unknown current_task: {current_task}. Must be 'vit_classification' or 'resnet_classification'")

ensure_directory(processed_dir)
ensure_directory(results_dir)

#  -------------------------------------------------------------------------------------------------

# Create transform functions using the processor
train_transform = make_train_transform(processor)
test_transform = make_test_transform(processor)

# Create compute_metrics function with results_dir closure
def compute_metrics_with_results_dir(eval_pred):
    """Wrapper for compute_metrics with results_dir"""
    return compute_metrics(eval_pred, results_dir)

#  -------------------------------------------------------------------------------------------------
# MAIN
#  -------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Detect device and set mixed precision flags
    device, use_fp16, use_bf16 = get_device()

    train_dataset, val_dataset, label_encoder = prepare_datasets(raw_dir, processed_dir, train_transform, test_transform)
    joblib.dump(label_encoder, f"{results_dir}/label_encoder.joblib")

    #  -------------------------------------------------------------------------------------------------

    labels = label_encoder.classes_

    model = model_class.from_pretrained(
        model_name_or_path,
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)},
        ignore_mismatched_sizes=True
    )

    # Move model to the detected device
    model = model.to(device)

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze only the classification head (last layer)
    for param in model.classifier.parameters():
        param.requires_grad = True

    print_trainable_parameters(model)

    #  -------------------------------------------------------------------------------------------------

    training_args = TrainingArguments(
        output_dir=results_dir,
        per_device_train_batch_size=16,
        eval_strategy="steps",
        num_train_epochs=50,
        fp16=use_fp16,  # Only True for CUDA
        bf16=use_bf16,  # True for MPS, False for others
        save_steps=500,
        eval_steps=500,
        logging_steps=500,
        learning_rate=2e-4,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to='none',
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics_with_results_dir,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor,
    )

    train_results = trainer.train()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_model()
    trainer.save_state()


    metrics = trainer.evaluate(val_dataset)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
#  -------------------------------------------------------------------------------------------------

