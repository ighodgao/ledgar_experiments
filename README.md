# ledgar_experiments

Create a .env file with your `OPENAI_API_KEY` and `ORGANIZATION_ID` and run:

`python3 eval.py --model_name=<model_name.` to run evaluation for models in ['gpt-3', 'roberta-base'. 'roberta-finetuned, 'tf-idf'].

If using `roberta-finetuned`, must also include  `--model_path` to saved checkpoint.

Run training job via:

det e create clm-config.yaml .
