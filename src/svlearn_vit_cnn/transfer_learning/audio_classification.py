
#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2024.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#   Use is limited to the duration and purpose of the training at SupportVectors.
#
#   Author: SupportVectors AI Training Team
#  -------------------------------------------------------------------------------------------------
#/Users/sangeetha/Supportvector2026/llm/projects/labs/vision_transformers_vs_cnn/docs/data/raw/family_voice

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
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)

#  -------------------------------------------------------------------------------------------------

# configurations: gunshot vs non-gunshot classification using LogMel spectrogram images
current_task = config['current_task']
gunshot_config = config['family-voice-dataset']
raw_dir = gunshot_config['path']
processed_dir = gunshot_config['processed_dir']

# Determine task-specific settings based on current_task
if current_task == 'vit_classification':
    results_dir = gunshot_config['results_vit']
    model_name_or_path = config['vision-transformer']['model_name']
    processor = ViTImageProcessor.from_pretrained(model_name_or_path)
    model_class = ViTForImageClassification
elif current_task == 'resnet_classification':
    results_dir = gunshot_config['results_cnn']
    model_name_or_path = config['cnn']['model_name']
    processor = AutoImageProcessor.from_pretrained(model_name_or_path)
    model_class = ResNetForImageClassification
else:
    raise ValueError(f"Unknown current_task: {current_task}. Must be 'vit_classification' or 'resnet_classification'")

ensure_directory(processed_dir)
ensure_directory(results_dir)

#  -------------------------------------------------------------------------------------------------

# Create transform functions using the processor (mel spectrograms are loaded as images)
train_transform = make_train_transform(processor)
test_transform = make_test_transform(processor)


def compute_metrics_with_results_dir(eval_pred):
    """Wrapper for compute_metrics with results_dir"""
    return compute_metrics(eval_pred, results_dir)


#  -------------------------------------------------------------------------------------------------

class EarlyStoppingOnF1Callback(TrainerCallback):
    """Stop training when eval F1 score reaches or exceeds the target."""

    def __init__(self, target_f1: float):
        self.target_f1 = target_f1

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        if metrics is None:
            return control
        eval_f1 = metrics.get("eval_f1")
        if eval_f1 is not None and eval_f1 >= self.target_f1:
            print(f"\nEarly stopping: eval_f1 {eval_f1:.4f} >= target {self.target_f1}")
            control.should_training_stop = True
        return control


#  -------------------------------------------------------------------------------------------------
# MAIN
#  -------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    device, use_fp16, use_bf16 = get_device()

    train_dataset, val_dataset, label_encoder = prepare_datasets(
        raw_dir, processed_dir, train_transform, test_transform
    )
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

    model = model.to(device)

    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the classifier (as before)
    for param in model.classifier.parameters():
        param.requires_grad = True

    # NEW: Unfreeze the last encoder block (Layer 11) for specialization
    if current_task == 'vit_classification':
        for param in model.vit.encoder.layer[-1].parameters():
            param.requires_grad = True
    elif current_task == 'resnet_classification':
        # Unfreeze the last ResNet stage
        for param in model.resnet.encoder.stages[-1].parameters():
            param.requires_grad = True

    print_trainable_parameters(model)

    #  -------------------------------------------------------------------------------------------------

    target_f1 = gunshot_config.get("target_f1_score")
    callbacks = []
    if target_f1 is not None:
        callbacks.append(EarlyStoppingOnF1Callback(target_f1=float(target_f1)))

    training_args = TrainingArguments(
        output_dir=results_dir,
        per_device_train_batch_size=16,
        eval_strategy="steps",
        num_train_epochs=50,
        fp16=use_fp16,
        bf16=use_bf16,
        save_steps=500,
        eval_steps=500,
        logging_steps=500,
        learning_rate=1e-4,
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
        processing_class=processor,
        callbacks=callbacks,
    )

    train_results = trainer.train()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_model()
    trainer.save_state()

    model.save_pretrained(f"{results_dir}/final_model")
    processor.save_pretrained(f"{results_dir}/final_model")
    joblib.dump(label_encoder, f"{results_dir}/final_model/label_encoder.joblib")

    metrics = trainer.evaluate(val_dataset)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
#  -------------------------------------------------------------------------------------------------
