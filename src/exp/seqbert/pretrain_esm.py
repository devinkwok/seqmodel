import sys
sys.path.append('./src')
import torch

from exp.seqbert import TOKENS_BP_IDX, TOKENS_BP
from exp.seqbert.pretrain import Pretrain
from esm.model import ProteinBertModel
from exp.seqbert.model import main


class ArgsObject(object):
    def __init__(self, args):
        self.__dict__ = args

class ESMBertWrapper(ProteinBertModel):

    tokens = TOKENS_BP

    def forward(self, tokens, no_attention):
        # ignore no_attention, as ESM already takes that into account
        n_layers = len(self.layers)
        result = super().forward(tokens, repr_layers=[0,n_layers], need_head_weights=False, return_contacts=False)
        predicted = result['logits'].transpose(1, 2)
        latent = result['representations'][n_layers].transpose(1, 2)
        embedded = result['representations'][0].transpose(1, 2)
        return predicted, latent, embedded

class PretrainESM(Pretrain):

    """Wrapper for ESM Bert tokens.
    """
    class DNAalphabet():
        tokens = TOKENS_BP_IDX
        padding_idx = TOKENS_BP_IDX['n']
        mask_idx = TOKENS_BP_IDX['n']
        cls_idx = TOKENS_BP_IDX['~']
        eos_idx = TOKENS_BP_IDX['/']
        prepend_bos = False  # only relevant for contact prediction head
        append_eos = False  # only relevant for contact prediction head

        def __len__(self):
            return len(self.tokens)

    """Wrapper for ESM Bert model.
        hparams: `ArgumentParser` object containing all hyperparameters
    """
    def __init__(self, **hparams):
        super().__init__(**hparams)
        args = {}
        args['arch'] = 'roberta_large'
        args['embed_dim'] = hparams['n_dims']
        args['ffn_embed_dim'] = hparams['feedforward_dims']
        args['attention_heads'] = hparams['n_heads']
        args['layers'] = hparams['n_layers']
        args['max_positions'] = hparams['seq_len']
        args['final_bias'] = True
        args['dropout'] = hparams['dropout']
        args['attention_dropout'] = hparams['dropout']
        args['activation_dropout'] = hparams['dropout']
        args['max_tokens'] = hparams['seq_len']
        self.model = ESMBertWrapper(ArgsObject(args), self.DNAalphabet())
        print('ESM model', self.model.model_version)


"""Run BERT pretraining using common `main` function to set up pl.
"""
if __name__ == '__main__':
    main(PretrainESM, PretrainESM)
