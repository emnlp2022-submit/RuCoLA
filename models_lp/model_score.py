import sys
import argparse
import torch
import math
import pickle
import csv
import numpy as np
import sentencepiece
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer, BertForMaskedLM, XLMRobertaTokenizer, XLMRobertaForMaskedLM, GPT2Tokenizer, GPT2LMHeadModel
from scipy.special import softmax


###########
#functions#
###########
def model_score(tokenize_input, model, tokenizer, device, args):


    if args.model_name.find("gpt") > 0:

        #prepend the sentence with <|endoftext|> token, so that the loss is computed correctly
        #tokenize_input += ['<|endoftext|>']
        tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)], device=device)
        labels = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)], device=device)
        labels[:,:1] = -1
        loss = model(tensor_input, labels=tensor_input)

        return float(loss[0]) * -1.0 * len(tokenize_input)

    elif args.model_name.startswith("xlm"):

        tokenize_combined = ["/s"] + tokenize_input + ["s"]
        lp = 0

        for i in range(len(tokenize_input)):
            # Mask a token that we will try to predict back with `BertForMaskedLM`
            masked_index = i + 1
            tokenize_masked = tokenize_combined.copy()
            tokenize_masked[masked_index] = '<mask>'

            # Convert token to vocabulary indices
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenize_masked)

            batched_indexed_tokens = []
            batched_indexed_tokens.append(indexed_tokens)
            tokens_tensor = torch.tensor(batched_indexed_tokens, device=device)

            # Models predictions
            with torch.no_grad():
                outputs = model(tokens_tensor)[0]
                predictions = outputs.cpu()

            # Calculate log-probabilities
            predicted_score = predictions[0, masked_index]
            predicted_prob = softmax(predicted_score.cpu().numpy())
            lp += np.log(predicted_prob[tokenizer.convert_tokens_to_ids([tokenize_combined[masked_index]])[0]])

            del tokens_tensor
            
        return lp

    elif args.model_name.startswith("bert"):

        tokenize_combined = ["[CLS]"] + tokenize_input + ["[SEP]"]

        lp = 0

        for i in range(len(tokenize_input)):
            # Mask a token that we will try to predict back with `BertForMaskedLM`
            masked_index = i + 1
            tokenize_masked = tokenize_combined.copy()
            tokenize_masked[masked_index] = '[MASK]'

            # Convert token to vocabulary indices
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenize_masked)
            segment_ids = [0]*len(tokenize_masked)

            batched_indexed_tokens = []
            batched_segment_ids = []

            batched_indexed_tokens.append(indexed_tokens)
            batched_segment_ids.append(segment_ids)

            tokens_tensor = torch.tensor(batched_indexed_tokens, device=device)
            segment_tensor = torch.tensor(batched_segment_ids, device=device)

            # Models predictions
            with torch.no_grad():
                outputs = model(tokens_tensor, token_type_ids=segment_tensor)[0]
                predictions = outputs.cpu()

            # Calculate log-probabilities
            predicted_score = predictions[0, masked_index]
            predicted_prob = softmax(predicted_score.cpu().numpy())
            lp += np.log(predicted_prob[tokenizer.convert_tokens_to_ids([tokenize_combined[masked_index]])[0]])

            del tokens_tensor
            del segment_tensor

        return lp


######
#main#
######
def main(args):

    #sentences
    sentencexdata = pd.read_csv(args.input_csv, sep="\t", usecols=['sentence'])

    #unigram frequencies
    if args.model_name.find("gpt") == -1:
        unigram_freq = pickle.load(open(args.unigram_pickle, "rb"))
        unigram_total = sum(unigram_freq.values()) 


    #Load pre-trained model and tokenizer
    if args.model_name.find("gpt") > 0:
        model = GPT2LMHeadModel.from_pretrained(args.model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
        
    elif args.model_name.startswith("bert"):
        model = BertForMaskedLM.from_pretrained(args.model_name)
        tokenizer = BertTokenizer.from_pretrained(args.model_name,
            do_lower_case=(True if "uncased" in args.model_name else False))
            
    elif args.model_name.startswith("xlm"):
        model = XLMRobertaForMaskedLM.from_pretrained(args.model_name)
        tokenizer = XLMRobertaTokenizer.from_pretrained(args.model_name,
            do_lower_case=(True if "uncased" in args.model_name else False))
    else:
        print("Supported models: gpt, bert and xlm-roberta only.")
        raise SystemExit

    #put model to device (GPU/CPU)
    device = torch.device(args.device)
    model.to(device)

    #eval mode; no dropout
    model.eval()
    
    #file for results    
    with open(args.output_csv, "w") as f:
        f.write('sent_id, lp, mean_lp, pen_lp, div_lp, sub_lp, slor, pen_slor\n')
        
    #loop through each sentence and compute system scores
    sent_total = 0
    for sent_id in tqdm(range(sentencexdata.shape[0])):

        text = sentencexdata['sentence'][sent_id]
        #uppercase first character
        text = text[0].upper() + text[1:]
        tokenize_input = tokenizer.tokenize(text)
        text_len = len(tokenize_input)

        tokenize_context = None
        
        #compute sentence logprob
        lp = model_score(tokenize_input, model, tokenizer, device, args)
        

        #unigram logprob
        if args.model_name.find("gpt") == -1:
            uni_lp = 0.0
            for w in tokenize_input:
                uni_freq = 1
                if w in unigram_freq:
                    uni_freq = max(1, float(unigram_freq[w]))
                uni_lp += math.log(uni_freq/unigram_total)

            #acceptability measures
            penalty = ((5+text_len)**0.8 / (5+1)**0.8)
            mean_lp = lp/text_len
            pen_lp = lp / penalty 
            div_lp = -lp / uni_lp
            sub_lp = lp - uni_lp
            slor = (lp - uni_lp) / text_len
            pen_slor = (lp - uni_lp) / penalty
            sent_id = sent_id
            
        else:
            #acceptability measures
            penalty = ((5+text_len)**0.8 / (5+1)**0.8)
            mean_lp = lp/text_len
            pen_lp = lp / penalty 
            div_lp = -1
            sub_lp = -1
            slor = -1
            pen_slor = -1
            sent_id = sent_id

        sent_total += 1
        
        with open(args.output_csv, "a") as f:
            f.write(f'{sent_id}, {lp}, {mean_lp}, {pen_lp}, {div_lp}, {sub_lp}, {slor}, {pen_slor}\n')




if __name__ == "__main__":

    #parser arguments
    desc = "Computes language acceptability using pytorch models"
    parser = argparse.ArgumentParser(description=desc)

    #arguments
    parser.add_argument("-i", "--input-csv", required=True, help="input csv file containing sentence data")
    parser.add_argument("-m", "--model-name", required=True,
        help="Pretrained model name (bert-base-multilingual-cased/xlm-roberta-base/gpt2")
    parser.add_argument("-u", "--unigram-pickle", help="Pickle file containing unigram frequencies (used for SLOR and NormLP)")
    parser.add_argument("-o", "--output-csv", required=True, help="Output csv with results")
    parser.add_argument("-d", "--device", default="cpu",
        help="specify the device to use (cpu or cuda:X for gpu); default=cpu")
    
    args = parser.parse_args()

    main(args)

