#!/usr/bin/env python3
"""Train local finite-answer LoRA/QLoRA adapters; benchmark without free decoding."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
from pathlib import Path

from minimol.decision_lora import (
    SPLITS,
    CompletionCollator,
    baseline_predictions,
    encode_example,
    fingerprint,
    fit_temperature,
    generate_policy_benchmark,
    load_records,
    option_probabilities,
    rescale_probabilities,
    score_predictions,
)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def load_base(args, training=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    local = Path(args.model).is_dir()
    if not local and not re.fullmatch(r'[0-9a-f]{40}', args.revision or ''):
        raise ValueError('Hub models require --revision with an immutable 40-character commit SHA')
    kwargs = {'revision': args.revision, 'trust_remote_code': False, 'local_files_only': local}
    tokenizer = AutoTokenizer.from_pretrained(args.model, **kwargs)
    if not tokenizer.chat_template:
        raise ValueError('a documented chat template is required')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    cuda = torch.cuda.is_available()
    if args.qlora and not cuda:
        raise ValueError('QLoRA requires CUDA in this runner')
    dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else torch.float16 if cuda else torch.float32
    if cuda:
        torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))
    kwargs['torch_dtype'] = dtype
    if args.qlora:
        kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
        kwargs['device_map'] = {'': int(os.environ.get('LOCAL_RANK', '0'))}
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    if cuda and not args.qlora:
        model.to(torch.cuda.current_device())
    model.config.use_cache = not training
    return model, tokenizer, dtype


def train(args):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import Trainer, TrainingArguments, set_seed

    records = load_records(args.data, required_splits=('train', 'development'))
    train_rows = [r for r in records if r.split == 'train']
    dev_rows = [r for r in records if r.split == 'development']
    out = Path(args.out)
    signature = {key: value for key, value in vars(args).items() if key not in {'func', 'resume'}}
    signature.update(data_sha256=fingerprint(args.data),
        runner_sha256=fingerprint(__file__), core_sha256=fingerprint(Path(__file__).parents[1] / 'src/minimol/decision_lora.py'))
    if args.resume:
        if not (out / 'run.json').exists() or json.loads((out / 'run.json').read_text())['signature'] != signature:
            raise ValueError('resume requires the original run directory and unchanged run signature')
        if Path(args.resume).resolve().parent != out.resolve():
            raise ValueError('resume checkpoint must belong to the output directory')
    elif out.exists() and any(out.iterdir()):
        raise ValueError('refusing to reuse nonempty run directory')
    set_seed(args.seed)
    model, tokenizer, dtype = load_base(args, training=True)
    if args.qlora:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    # Also allows gradient flow through frozen embeddings for non-quantized LoRA.
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.rank * 2,
        lora_dropout=.05, target_modules='all-linear', bias='none', task_type='CAUSAL_LM', use_rslora=args.rslora))
    train_data = [encode_example(tokenizer, r, args.max_length) for r in train_rows]
    dev_data = [encode_example(tokenizer, r, args.max_length) for r in dev_rows]
    training_args = TrainingArguments(output_dir=str(out), num_train_epochs=args.epochs,
        max_steps=args.max_steps, per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1, gradient_accumulation_steps=args.accumulation,
        learning_rate=args.lr, warmup_ratio=.05, lr_scheduler_type='cosine', weight_decay=.01,
        bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        eval_strategy='epoch', save_strategy='epoch', save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model='eval_loss', greater_is_better=False,
        ddp_find_unused_parameters=False, report_to=[], logging_steps=10,
        seed=args.seed, data_seed=args.seed, push_to_hub=False)
    trainer = Trainer(model=model, args=training_args, train_dataset=train_data,
                      eval_dataset=dev_data, data_collator=CompletionCollator(tokenizer.pad_token_id))
    if trainer.is_world_process_zero():
        versions = {n: importlib.metadata.version(n) for n in ['torch', 'transformers', 'peft', 'accelerate']}
        write(out / 'run.json', {'signature': signature, 'versions': versions,
            'base_resolved_commit': getattr(model.config, '_commit_hash', None),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'loaded_parameters': sum(p.numel() for p in model.parameters()),
            'quantization': 'nf4' if args.qlora else 'none',
            'splits_used_for_fit': ['train', 'development'], 'test_used': False,
            'tracks': sorted({r.track for r in records}),
            'license_statuses': sorted({r.license_status for r in records})})
        tokenizer.save_pretrained(out / 'adapter')
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(str(out / 'adapter'))
    if trainer.is_world_process_zero():
        write(out / 'training_summary.json', {'best_checkpoint': trainer.state.best_model_checkpoint,
            'best_development_completion_loss': trainer.state.best_metric,
            'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
            'note': 'Completion loss selects adapters; task metrics require benchmark prediction/evaluation.'})


def predict(args):
    import torch
    from peft import PeftModel

    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('prediction runner is single-process; do not launch with torchrun')
    records = [r for r in load_records(args.data) if r.split == args.split]
    if not records:
        raise ValueError('empty prediction split')
    if Path(args.out).exists():
        raise ValueError('refusing to overwrite predictions')
    model, tokenizer, _ = load_base(args)
    if args.adapter:
        # Refuse an accidentally different tokenizer/base than used for adapter training.
        run_path = Path(args.adapter).parent / 'run.json'
        if not run_path.exists():
            raise ValueError('adapter requires adjacent original run.json')
        signature = json.loads(run_path.read_text())['signature']
        if signature['model'] != args.model or signature['revision'] != args.revision:
            raise ValueError('adapter base model/revision mismatch')
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    rows = []
    for record in records:
        probabilities = option_probabilities(model, tokenizer, record, args.max_length)
        rows.append({'id': record.id, 'codes': [o.code for o in record.options], 'probabilities': probabilities,
                     'selected_value': record.options[max(range(len(probabilities)), key=probabilities.__getitem__)].value})
    write(args.out, {'data_sha256': fingerprint(args.data), 'split': args.split,
        'model': args.model, 'revision': args.revision, 'adapter': args.adapter,
        'adapter_sha256': fingerprint(Path(args.adapter) / 'adapter_model.safetensors') if args.adapter else None,
        'temperature': 1., 'confidence_status': 'uncalibrated restricted candidate likelihood',
        'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        'predictions': rows})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    build = commands.add_parser('build-demo')
    build.add_argument('--out', required=True)
    build.add_argument('--groups', type=int, default=400)
    build.add_argument('--seed', type=int, default=11)
    validate = commands.add_parser('validate')
    validate.add_argument('--data', required=True)
    for command in ['train', 'predict']:
        p = commands.add_parser(command)
        p.add_argument('--data', required=True)
        p.add_argument('--model', default='Qwen/Qwen3-0.6B')
        p.add_argument('--revision')
        p.add_argument('--out', required=True)
        p.add_argument('--max-length', type=int, default=2048)
        p.add_argument('--qlora', action='store_true')
        if command == 'train':
            p.add_argument('--rank', type=int, default=16)
            p.add_argument('--rslora', action='store_true')
            p.add_argument('--epochs', type=int, default=3)
            p.add_argument('--batch-size', type=int, default=1)
            p.add_argument('--accumulation', type=int, default=16)
            p.add_argument('--lr', type=float, default=1e-4)
            p.add_argument('--seed', type=int, default=11)
            p.add_argument('--max-steps', type=int, default=-1)
            p.add_argument('--resume')
        else:
            p.add_argument('--adapter')
            p.add_argument('--split', choices=SPLITS, default='development')
    baseline = commands.add_parser('baseline')
    baseline.add_argument('--data', required=True)
    baseline.add_argument('--out', required=True)
    baseline.add_argument('--kind', choices=['uniform', 'policy'], required=True)
    baseline.add_argument('--split', choices=SPLITS, default='development')
    calibration = commands.add_parser('calibrate')
    calibration.add_argument('--data', required=True)
    calibration.add_argument('--predictions', required=True)
    calibration.add_argument('--out', required=True)
    score = commands.add_parser('score')
    score.add_argument('--data', required=True)
    score.add_argument('--predictions', required=True)
    score.add_argument('--out', required=True)
    score.add_argument('--calibration')
    args = parser.parse_args()
    if args.command == 'build-demo':
        print(json.dumps(generate_policy_benchmark(args.out, args.groups, args.seed)))
    elif args.command == 'validate':
        records = load_records(args.data, required_splits=SPLITS)
        print(json.dumps({'records': len(records), 'sha256': fingerprint(args.data),
                          'groups': len({r.group_id for r in records})}))
    elif args.command == 'baseline':
        records = [r for r in load_records(args.data) if r.split == args.split]
        if not records:
            raise ValueError('empty baseline split')
        predictions = baseline_predictions(records, args.kind)
        write(args.out, {'data_sha256': fingerprint(args.data), 'split': args.split,
                         'baseline': args.kind, 'predictions': predictions,
                         'metrics': score_predictions(records, predictions)})
    elif args.command == 'train':
        if min(args.rank, args.epochs, args.batch_size, args.accumulation, args.max_length) < 1 or args.lr <= 0:
            parser.error('training counts and learning rate must be positive')
        train(args)
    elif args.command == 'predict':
        predict(args)
    else:
        artifact = json.loads(Path(args.predictions).read_text())
        if artifact['data_sha256'] != fingerprint(args.data):
            raise ValueError('benchmark hash mismatch')
        records = [r for r in load_records(args.data) if r.split == artifact['split']]
        identity = {k: artifact.get(k) for k in ['model', 'revision', 'adapter_sha256']}
        if args.command == 'calibrate':
            temperature = fit_temperature(records, artifact['predictions'])
            write(args.out, {'temperature': temperature, 'data_sha256': artifact['data_sha256'],
                             'identity': identity, 'fit_split': 'calibration',
                             'predictions_sha256': fingerprint(args.predictions)})
            return
        if args.calibration:
            calibration = json.loads(Path(args.calibration).read_text())
            if (calibration['fit_split'] != 'calibration' or calibration['identity'] != identity
                    or calibration['data_sha256'] != artifact['data_sha256']):
                raise ValueError('calibration model or data identity mismatch')
            for pred in artifact['predictions']:
                pred['probabilities'] = rescale_probabilities(pred['probabilities'], calibration['temperature'])
        write(args.out, {'split': artifact['split'], 'metrics': score_predictions(records, artifact['predictions']),
                         'scope': 'scores are separated by track; synthetic policy is not biological validation'})


if __name__ == '__main__':
    main()
