#! /usr/bin/env python3
# coding=utf-8

import os
import sys
import argparse
from tqdm import trange

import torch
import torch.optim
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

lab_root = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..')
sys.path.insert(1, lab_root)
from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer

from IPython import embed

def top_k_logits(logits, k, probs=False):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        if probs:
            return torch.where(logits < batch_mins, torch.ones_like(logits) * 0.0, logits)
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -1e10, logits)

def sample_sequence(model, length, start_token=None, batch_size=None, context=None, temperature=1,
                    top_k=0, device='cuda', sample=True, return_past=False):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
        context = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = torch.full((batch_size, 1), start_token, device=device, dtype=torch.long)
    # context.requires_grad_()=True
    prev = context
    output = context
    past = None
    with torch.no_grad():
        for i in trange(length, ascii=True):
            logits, past = model(prev, past=past)
            logits = logits[:, -1, :] / temperature
            logits = top_k_logits(logits, k=top_k) # do nothing if k=0
            log_probs = F.softmax(logits, dim=-1)
            if sample:
                prev = torch.multinomial(log_probs, num_samples=1)
            else:
                _, prev = torch.topk(log_probs, k=1, dim=-1)
            # prev is the next character, past is something [2, 1, 16, x, 64] where x grows from 1 to length
            # embed()
            # print('sample sequence {}: prev shape {} past shape {}'.format(i,
            # list(prev[0].size()), list(past[0].size())))
            output = torch.cat((output, prev), dim=1)
            #print(output)
    if return_past:
        return output, past
    else:
        return output


def sample_from_hidden(model, length, hidden, context=None, past=None, temperature=1,
                       top_k=0, device='cuda', sample=True, noise_level=1e-1):
    output = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0) if context else None
    with torch.no_grad():
        for i in trange(length, ascii=True):
            logits = model.forward_hidden(hidden)
            logits = logits[:, -1, :] / temperature
            logits = top_k_logits(logits, k=top_k) # do nothing if k=0
            log_probs = F.softmax(logits, dim=-1)
            if sample:
                prev = torch.multinomial(log_probs, num_samples=1)
            else:
                _, prev = torch.topk(log_probs, k=1, dim=-1)
            # prev is the next character, past is something [2, 1, 16, x, 64] where x grows from 1 to length
            #embed()
            #print('sample sequence {}: prev shape {} past shape {}'.format(i, list(prev[0].size()), list(past[0].size())))
            output = prev if output is None else torch.cat((output, prev), dim=1) # update output
            if i == 0:
                _, past = model(output, past=None)      # update past. Take the whole input context
            else:
                _, past = model(prev, past=past)        # update past. Take one next token
            hidden = model.hidden_states            # update hidden
            #print('output', output)
            #print('hidden', hidden)
            
            # do something with the hidden
            hidden = modify_hidden(hidden, noise_level)
    return output

def modify_hidden(input_tensor, noise_level=1e-1):
    # input_tensor shape: (1, 1, length)
    length = input_tensor.shape[-1]
    ret = input_tensor + torch.rand(length).cuda() * noise_level
    return ret

def compute_log_likelihood(model, phrase, tokenizer, device):
    token_ids = tokenizer.encode(phrase)
    batch_size = 1
    context = torch.tensor(token_ids, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    print("Computing LL of phrase \"{}\"".format(phrase))
    print("After encoding, number of tokens {}".format(len(token_ids)))
    with torch.no_grad():
        logits, past = model(context, past=None)

    _idxs = range(len(token_ids) - 1)
    token_ids = token_ids[1:]
    logits = logits[0, :-1]

    probs = F.softmax(logits, dim=-1)
    likelihoods = probs[_idxs, token_ids]
    assert len(list(likelihoods.shape)) == 1

    log_likelihoods = torch.log(likelihoods)
    ll_list = [ls.item() for ls in log_likelihoods]
   
    for token, llh in zip(token_ids, log_likelihoods):
        print("LL of token   {} (\'{}\')  ==>  {:.4f}".format(token, tokenizer.decode([token]), llh))

    print("LL of the phrase (sum of the above): {}".format(np.sum(ll_list)))
    return np.sum(ll_list)


def get_embedding_grad(model, enc, context=None, target=40, device='cuda', ll_only=False, opt_embed=False):
    assert context is not None, 'Input text is needed'
    # context = Variable(torch.tensor(context, device=device, dtype=torch.float),
    #                    requires_grad=True).unsqueeze(0)#.repeat(1, 1)
    
    context = torch.tensor(context, device=device, dtype=torch.float).unsqueeze(0)
    
    model.zero_grad()
    logits, past = model(context, past=None)
    
    # make sure it is the same as above
    # logits_1, past_1 = model.forward_embed(model.transformer.i_embeds, past=None)

    logits = logits[:, -1, :]
    log_probs = F.softmax(logits, dim=-1)

    if len(target) > 1:
        nll = sum([-torch.log(log_probs[:, tar]) for tar in target])
    else:
        nll = - torch.log(log_probs[:, target])

    
    with torch.no_grad():
        # logits = top_k_logits(logits, k=1) # do nothing if k=0
        log_probs = F.softmax(logits, dim=-1)
        top1, top1ind = torch.topk(log_probs, k=1, dim=-1)
    
        print('LL of target : {}'.format(-nll.data.squeeze().cpu().numpy()))
        print('LL of top 1 : {}'.format(torch.log(top1).data.squeeze().cpu().numpy()))

    if ll_only:
        return
   
    if opt_embed:  # optimizin in embedding space
        orig_embed = model.transformer.i_embeds.clone()
        embed_vars = Variable(model.transformer.i_embeds, requires_grad=True)
        # optimizer = torch.optim.SGD([embed_vars], lr=0.01, momentum=0.9)
        optimizer = torch.optim.Adam([embed_vars], lr=0.01)
        optimizer.zero_grad()
        
        for ss in range(50):
            # nll.backward(retain_graph=True)
            nll.backward()
            optimizer.step()
            
            logits, past = model.forward_embed(embed_vars, past=None)
            logits = logits[:, -1, :]
            log_probs = F.softmax(logits, dim=-1)

            if len(target) > 1:
                nll = sum([-torch.log(log_probs[:, tar]) for tar in target])
            else:
                nll = - torch.log(log_probs[:, target])
            
            print('LL of target (step {}): {}'.format(ss, -nll.data.squeeze().cpu().numpy()))
            # print('Sanity check: embed_vars sum: {}'.format(embed_vars.sum().cpu().detach().numpy()))

    
        # searching in token space
        output_ids = torch.empty_like(context.long())
        with torch.no_grad():
            all_embeds = model.transformer.wte.weight   # [50257, 1024]
            embed_vars_unbind = torch.unbind(embed_vars, dim=1)
            orig_embed_unbind = torch.unbind(orig_embed, dim=1)

            cc = 0
            for ie_new, ie_orig, orig_id in zip(embed_vars_unbind, orig_embed_unbind, context.squeeze(0)):
                new_id = (all_embeds - ie_new).abs().sum(1).argmin()

                print('emb {}: {} (`{}`) to {} (`{}`)'.format(cc, orig_id.tolist(), enc.decode([orig_id.tolist()]),
                        new_id.tolist(), enc.decode([new_id.tolist()])))

                output_ids[0, cc] = new_id
                cc += 1
            
        output_ids = torch.cat((context.long(), output_ids), dim=1)
    return output_ids 



    ## searching in token space
    # model.transformer.i_embeds.retain_grad()
    # nll.backward()
    # step = 0.01
    #
    ## with torch.no_grad():
    # if True:
    #    input_grads = model.transformer.i_embeds.grad   # [batch, length, 1024]
    #    #input_grads = input_grads.squeeze(0)            # [length, 1024]
    #    input_embeds = model.transformer.i_embeds       # [batch, length, 1024]
    #    input_embeds_unbind = torch.unbind(input_embeds, dim=1)
    #    all_embeds = model.transformer.wte.weight   # [50257, 1024]
    #    
    #    opts = [torch.optim.Adam([Variable(ie, requires_grad=True)], lr=0.01) for ie in input_embeds_unbind]
    #    

    #    ## HERE
    #    # for ss in range(50):
    #    #    input_embeds.data.sub_(step * input_grads.data)
    #    #    #input_embeds.data.add_(step * input_grads.data)
    #    #    
    #    #    logits, past = model.forward_embed(input_embeds, past=None)
    #    #    logits = logits[:, -1, :]
    #    #    log_probs = F.softmax(logits, dim=-1)

    #    #    if len(target) > 1:
    #    #        nll = sum([-torch.log(log_probs[:, tar]) for tar in target])
    #    #    else:
    #    #        nll = - torch.log(log_probs[:, target])
    #    #    
    #    #    print('LL of target (step {}): {}'.format(ss, -nll.data.squeeze().cpu().numpy()))
    #    #    
    #    # embed()

    #    search_order = input_grads.sum(-1).squeeze().abs().argsort(descending=True)

    #    output_ids = context.long()
    #    cc = 0
    #    #n_tokens_to_change = 1 
    #    for order, orig_id in zip(search_order,  context.squeeze(0)[search_order]):
    #        embed()
    #        
    #        ie = input_embeds_unbind[order]

    #        orig_id = orig_id.long()
    #        opt = opts[order]
    #        opt.zero_grad()

    #        new_id = abs(all_embeds - ie).sum(1).argmin().data # new_id == orig_id 

    #        #if cc < n_tokens_to_change:
    #        #    while new_id == orig_id: # 
    #                #ie.data.sub_(step * ig.data)
    #                #ie.data.add_(step * ig.data)
    #        for opt_step in range(50):
    #                opt.step()
    #                new_id = abs(all_embeds - ie).sum(1).argmin().data

    #        print('emb {}: {} (`{}`) to {} (`{}`)'.format(order, orig_id.tolist(), enc.decode([orig_id.tolist()]),
    #                new_id.tolist(), enc.decode([new_id.tolist()])))

    #        output_ids[0, order] = new_id

    #        #output_ids = torch.cat((output_ids, new_id.reshape(1,1)), dim=1)
    #        cc += 1

    #    output_ids = torch.cat((context.long(), output_ids), dim=1)
    # print(context.grad)
    return output_ids 

def run_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', '-M', type=str, default='gpt-2_pt_models/774M/', 
                        help='pretrained model name or path to local checkpoint')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--length", type=int, default=-1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument('--unconditional', action='store_true', help='If true, unconditional generation.')
    parser.add_argument('--nocuda', action='store_true', help='no cuda')
    parser.add_argument('--opt_ll', action='store_true', help='nll optimize')
    parser.add_argument('--get_ll', action='store_true', help='compute log likelihood of sentence')
    parser.add_argument('--hidden_playground', action='store_true', help='play around in the hidden representation')
    parser.add_argument("--noise_level", type=float, default=1e-1)
    parser.add_argument("--cond-text", type=str, default='', help='Prefix texts to condition on')
    parser.add_argument('--output', type=str, default=os.environ.get('GIT_RESULTS_MANAGER_DIR', None), help='output directory')
    args = parser.parse_args()
    print(args)

    if args.batch_size == -1:
        args.batch_size = 1
    assert args.nsamples % args.batch_size == 0

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.nocuda:
        device = torch.device("cpu") 

    print('device is {}'.format(device))

    enc = GPT2Tokenizer.from_pretrained(args.model_path)
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    
    if args.length == -1:
        args.length = model.config.n_ctx // 2
    elif args.length > model.config.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % model.config.n_ctx)

    #while True:
    generated = 0
    for _ in range(10):
        context_tokens = []
        if not args.unconditional:
            #raw_text = input("Model prompt >>> ")
            raw_text = args.cond_text
            while not raw_text:
                print('Prompt should not be empty!')
                raw_text = input("Model prompt >>> ")
            context_tokens = enc.encode(raw_text)
            for _ in range(args.nsamples // args.batch_size):
                out = sample_sequence(
                    model=model, length=args.length,
                    context=context_tokens,
                    start_token=None,
                    batch_size=args.batch_size,
                    temperature=args.temperature, top_k=args.top_k, device=device
                )
                #out = out[:, len(context_tokens):].tolist()
                out = out[:, 0:].tolist()
                for i in range(args.batch_size):
                    generated += 1
                    text = enc.decode(out[i])
                    print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                    print(text)
                    if args.output:
                        filepath = os.path.join(args.output, "generated_{}.txt".format(generated))
                        with open(filepath, "w") as f:
                            f.write(text)
                    
           # print("=" * 80)
        if args.unconditional:
          generated = 0
          for _ in range(args.nsamples // args.batch_size):
              out = sample_sequence(
                  model=model, length=args.length,
                  context=None,
                  start_token=enc.encoder['<|endoftext|>'],
                  batch_size=args.batch_size,
                  temperature=args.temperature, top_k=args.top_k, device=device
              )
              out = out[:,1:].tolist()
              for i in range(args.batch_size):
                  generated += 1
                  text = enc.decode(out[i])
                  print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                  print(text)
          #print("=" * 80)
          if args.unconditional:
              break


if __name__ == '__main__':
    run_model()


