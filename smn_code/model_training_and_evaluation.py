from typing import Tuple
import numpy as np
import argparse
from evaluate import load
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    AutoTokenizer,
)
import torch

# The code is partially based on the following codes:
# https://colab.research.google.com/gist/peregilk/3c5e838f365ab76523ba82ac595e2fcc/nbailab-finetuning-and-evaluating-a-bert-model-for-classification.ipynb#scrollTo=4utMn85m12vB
# https://github.com/doantumy/LM_for_Party_Affiliation_Classification
# It is very similar to the code for the NoReC dataset also used during the thesis (https://github.com/Karolill/master_thesis/blob/main/norec_code/model_training_and_evaluation.py) but some changes has been made so the code can also be used for AL. 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device is: {device}')


def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True, max_length=512)


# Create a function to evaluate the training. F1-score and precision for negative class is used. 
# They are loaded from a local folder because the VM will not allow me to download it from the Hugging Face hub.
f1_metric = load('./metrics/f1')
precision_metric = load('./metrics/precision')


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # It is more important to discover the negative classes than the positive ones, therefore the F1-score of the
    # negative class is being used.
    f1 = f1_metric.compute(
        predictions=predictions,
        references=labels,
        average='binary',
        pos_label=0,
    )

    # As it is not desirable to misclassify a lot of positive examples (but this seems to be an issue), the
    # precision of the negative class will also be returned. The lower this precision is, the more positive exampels
    # are misclassified.
    precision = precision_metric.compute(
        predictions=predictions,
        references=labels,
        average='binary',
        pos_label=0,
    )

    return {'f1_neg': f1['f1'], 'precision_neg': precision['precision']}


def create_and_train_model(
    model_path: str,
    model_name: str,
    training_data: Dataset,
    eval_data: Dataset = None,
    epochs: float = 3,
    learning_rate: float = 5e-5,
    warmup_ratio: float = 0.1,
    optimizer: str = 'adamw_hf',
    weight_decay: float = 0,
    al: bool = False,
    num_al_examples: int = 0,
    load_best_model_at_end: bool = True,
    evaluation_strategy: str = 'epoch',
) -> Tuple[str, str]:
    """
    Create a model based on a pretrained model, then fine-tune the model and save it to a file.
    Args:
        :param model_path: path that is used to load the pre-trained model (sometimes just the model name)
        :param model_name: chosen name of the model being trained
        :param training_data: training dataset (encoded) to use for fine-tuning
        :param eval_data: evaluation dataset (encoded) to use for evaluation during fine-tuning
        :param epochs: number of epochs to train the model
        :param learning_rate: the learning rate to use
        :param warmup_ratio: Ratio of total training steps used for a linear warmup from 0 to learning_rate
        :param optimizer: The optimizer to use. Either adamw_hf, adamw_torch, adamw_apex_fused, adamw_anyprecision or
        adafactor.
        :param weight_decay: The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights
        in AdamW optimizer.
        :param al: True if active learning is being done, False if it is only for normal model training. 
        :param num_al_examples: Numbers of examples currently being used for AL. Only useful if al=True. 
        :param load_best_model_at_end: Decide if the model from the best epoch should be loaded and saved in the end. If you are 
        doing AL, whatever is written here will be overwritten to False anyway, so this parameter is for regular model training. 
    :returns the path to the fine-tuned model
    """

    # Create the model by getting the pre-trained one
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2, force_download=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Specify training arguments so they can be sent to the trainer later. 
    # There will be different arguments for AL and normal training, as the processes are not exactly the same. 
    if al:
        output_dir = f'../training_output/trainer_ex{num_al_examples}'
        evaluation_strategy = 'no'
        load_best_model_at_end = False  # In AL it does not make sense to load anything but the final model. 
        save_strategy = 'no'
    else:
        output_dir = f'../training_output/trainer_{model_name}_LR{learning_rate}_WR{warmup_ratio}_OPTIM{optimizer}_WD{weight_decay}'
        evaluation_strategy = evaluation_strategy
        save_strategy = 'epoch'
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        optim=optimizer,
        evaluation_strategy=evaluation_strategy,
        num_train_epochs=epochs,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model='f1_neg',
        save_strategy=save_strategy,
    )

    if al:
        print(f'Starting to train {model_name} with {num_al_examples} training examples')
    else:
        print(f'Starting to train {model_name} with LR: {learning_rate}, WR: {warmup_ratio}, OPTIM: {optimizer} and WD: '
              f'{weight_decay}')

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_data,
        eval_dataset=eval_data,
        compute_metrics=compute_metrics,
    )

    # fine-tune model
    trainer.train()

    if al:
        print(f'Done training {model_name} with {num_al_examples} training examples')
    else:
        print(f'Done training {model_name} with LR: {learning_rate}, WR: {warmup_ratio}, OPTIM: {optimizer} and WD: '
              f'{weight_decay}')

    # save the model to a folder so it can be used later. If load_best_model_at_end=True, the model from the best
    # checkpoint will be saved
    if al:
        model_directory = f'./models/models_al/{model_name}_ex{num_al_examples}'
    else:
        model_directory = f'./models/{model_name}_LR{learning_rate}_WR{warmup_ratio}_' \
                          f'OPTIM{optimizer}_WD{weight_decay}'
    tokenizer.save_pretrained(model_directory)
    model.save_pretrained(model_directory)

    # Save information about loss, f1-score, runtime ++ for each epoch in a folder if AL=False. This information is 
    # not needed during AL because the labels are partially wrong anyway. 
    if not al:
        state_directory = f'./scores/state_{model_name}_LR{learning_rate}_WR{warmup_ratio}_' \
                          f'OPTIM{optimizer}_WD{weight_decay}'
        text_file = open(state_directory, 'w')
        text_file.write(str(trainer.state.log_history))
        text_file.close()

    if al:
        print(f'Done saving {model_name} with {num_al_examples} training examples')
    else:
        print(f'Done saving {model_name} with LR: {learning_rate}, WR: {warmup_ratio}, OPTIM: {optimizer} and WD: '
              f'{weight_decay}')
        # If not AL, then the path to the scores should also be returned alongside the model, to make automatic plotting of
        # of the scores possible later
        return model_directory, state_directory

    # return directory name of model so both model and tokenizer can be fetched later
    return model_directory