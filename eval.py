from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForSeq2SeqLM, RobertaForCausalLM, AutoConfig
from transformers import AutoTokenizer, RobertaTokenizer, RobertaModel
from transformers import pipeline, AutoTokenizer
from contract_provision_types import label_list
import numpy as np
import os
import openai
from dotenv import load_dotenv
from fuzzywuzzy import fuzz

class Evaluate():

    load_dotenv()

    openai.organization = os.getenv("ORGANIZATION_ID")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    def __init__(self):
        self.dataset = load_dataset("lex_glue", "ledgar")
        self.models = ["tf-idf", "gpt-3", "roberta-base"]
        self.max_tokens = None
        self.model_name = None
        self.tokenizer = None
        self.fuzzy_string_matching = False
        self.fuzzy_string_ratio = 75

        self.few_shot_examples = [
            {
                "provision": "Except as otherwise set forth in this Debenture, the Company, for itself and its legal representatives, successors and assigns, expressly waives presentment, protest, demand, notice of dishonor, notice of nonpayment, notice of maturity, notice of protest, presentment for the purpose of accelerating maturity, and diligence in collection.",
                "label": "Waivers",
            },
            {
                "provision": "No ERISA Event has occurred or is reasonably expected to occur that, when taken together with all other such ERISA Events for which liability is reasonably expected to occur, could reasonably be expected to result in a Material Adverse Effect. Neither Borrower nor any ERISA Affiliate maintains or contributes to or has any obligation to maintain or contribute to any Multiemployer Plan or Plan, nor otherwise has any liability under Title IV of ERISA.",
                "label": "Erisa",
            },
            {
                "provision": "This Amendment may be executed by one or more of the parties hereto on any number of separate counterparts, and all of said counterparts taken together shall be deemed to constitute one and the same instrument. This Amendment may be delivered by facsimile or other electronic transmission of the relevant signature pages hereof.",
                "label": "Counterparts",
            },
    
        ]

    def create_prompt(self, few_shot_examples, provision):
        prompt = ""
        for example in self.few_shot_examples:
            prompt += f"{example['provision']}\nLabel: {example['label']}\n===\n"
        prompt += f"\nWhat is the label of the following legal provision?\n{provision}\nLabel:"
        return prompt

    def compute_tfidf_labels(self):
        vectorizer = TfidfVectorizer(min_df=1, stop_words="english")
        vectorizer.fit([example["text"] for example in self.dataset["train"]])
        tfidf_docs = vectorizer.transform([example["text"] for example in self.dataset["test"]])
        tfidf_labels = vectorizer.transform(label_list)
        similarity_scores = tfidf_docs * tfidf_labels.T
        lf_votes = np.argmax(similarity_scores, axis=1)
        lf_votes = [i[0] for i in lf_votes]
        actual_labels = [example["label"] for example in self.dataset["test"]]

        correct = sum(a == b for a, b in zip(lf_votes, actual_labels))
        accuracy = correct / len(actual_labels)
        return accuracy 


    def evaluate(self, model_name):
        label_string = ", ".join(label_list)
        prompt = f"The possible labels are {label_string}.\n\n"

        for example in self.few_shot_examples:
            prompt += f"{example['provision']}\nLabel: {example['label']}\n===\n"
        
        correct = 0
        total = 0
        for provision, label in zip(self.dataset["test"]["text"], self.dataset["test"]["label"]):
            total +=1
            print(total)
            
            prompt = self.create_prompt(self.few_shot_examples, provision)
            prompt_enc = self.tokenizer.encode(prompt, truncation=False)
            
            if len(prompt_enc) > self.max_tokens:
                # Create a context window around the new prompt within the token limit
                new_prompt_start = prompt_enc.index(self.tokenizer.encode(f"\nWhat is the label of the following legal provision?\nProvision {provision}\nLabel:", add_special_tokens=False)[0])
                prompt_enc = prompt_enc[new_prompt_start-self.max_tokens:]

            prompt = self.tokenizer.decode(prompt_enc)
            
            if model_name in ["roberta-base", "google/flan-t5-xxl"]:        

                if model_name == "roberta-base":
                    config = AutoConfig.from_pretrained("roberta-base")
                    config.is_decoder = True
                    model = RobertaForCausalLM.from_pretrained("roberta-base", config=config)

                if model_name == "google/flan-t5-xxl":
                    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xxl")

                
                model = model.to("cuda")
                
                inputs = tokenizer(prompt, return_tensors='pt').input_ids.to("cuda")
                outputs = model.generate(inputs, max_length=self.max_tokens, num_return_sequences=1)
                predicted_label = tokenizer.decode(outputs[0]).split("Label:")[-1]
                print(predicted_label)

                if predicted_label.strip() == label_list[label]:
                    correct += 1


            elif model_name == "gpt-3":
                response = openai.Completion.create(
                    engine="text-davinci-002",
                    prompt=prompt,
                    temperature=0.5,
                    max_tokens=self.max_tokens,
                    top_p=1.0,
                )
                predicted_label = response['choices'][0]['text'].strip()
                if self.fuzzy_string_matching:
                    if fuzz.ratio(predicted_label == label_list[label]) > self.fuzzy_string_ratio:
                        correct += 1
                else:
                    if predicted_label == label_list[label]:
                        correct +=1
            
            if total == 2000:
                
        accuracy = correct/total
        return accuracy


if __name__ == "__main__":
    evaluator = Evaluate()

    for model_name in evaluator.models:
        evaluator.model_name = model_name
        
        accuracy = 0

        if model_name == "tf-idf":
            accuracy = evaluator.compute_tfidf_labels()

        elif model_name in ["roberta-base", "google/flan-t5-xxl"]:
            evaluator.tokenizer = AutoTokenizer.from_pretrained(model_name)
            evaluator.max_tokens = 512
            accuracy = evaluator.evaluate(model_name)         

        elif model_name == "gpt-3":
            evaluator.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            evaluator.max_tokens = 512
            accuracy = evaluator.evaluate(model_name)

        print(f"{model_name} Accuracy: {accuracy}")
