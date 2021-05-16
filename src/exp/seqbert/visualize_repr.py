import sys
sys.path.append('./src')
import os
import numpy as np
import torch
import esm.pretrained
import torch.nn.functional as F
from PIL import Image
from argparse import ArgumentParser
from pretrain import Pretrain
from pretrain_esm import PretrainESM


base_map = {
    0: 0xffff6969,
    1: 0xffb3ff69,
    2: 0xff69fdff,
    3: 0xffb769ff,
    4: 0xff000000,
}

def seq_to_color(seq, cmap):
    r = np.zeros(len(seq), dtype=np.uint8)
    g = np.zeros(len(seq), dtype=np.uint8)
    b = np.zeros(len(seq), dtype=np.uint8)
    for i, s in enumerate(seq):
        color = cmap.get(s.item(), 0xffffffff)
        r[i] = (color & 0x00ff0000) >> 16
        g[i] = (color & 0x0000ff00) >> 8
        b[i] = color & 0x000000ff
    return np.stack([r, g, b], axis=1)

def scale_to_color(matrix):  # assume matrix elements in (-1, 1)
    # if less than 0, red, otherwise blue
    pos = np.zeros_like(matrix)
    neg = np.zeros_like(matrix)
    pos[matrix > 0] = matrix[matrix > 0]
    neg[matrix < 0] = matrix[matrix < 0]
    r = np.ones_like(pos)
    g = np.ones_like(pos)
    b = np.ones_like(pos)
    r = np.uint8(np.round((r - pos) * 255))
    g = np.uint8(np.round((g - pos - neg) * 255))
    b = np.uint8(np.round((b - neg) * 255))
    return np.stack([r, g, b], axis=2)

def att_to_color(matrix):  # assume matrix elements in (0, 1), head x H x W
    # use hsv, hue is max attention head, sat is max attention, value is mean log attention
    # log transform probabilities to get more linear scale, cutoff at -255
    n_heads = matrix.shape[0]
    epsilon = 1e-16
    min_val = -1 * np.log(epsilon)
    log_prob = (np.log(matrix + epsilon) + min_val) / min_val * 255
    h = np.uint8(np.round(np.argmax(matrix, axis=0) / n_heads * 255))
    s = np.uint8(np.round(np.max(matrix, axis=0) * 255))
    v = np.uint8(np.round(np.mean(log_prob, axis=0)))
    return np.stack([h, s, v], axis=2)

def results_to_imgs(result, filename, seq_img=None, use_esm=False):
    # for each repr, create cosine similarity matrix
    for k, v in result['representations'].items():
        # shape of representation is batch x seq_len x dims, convert to batch x dims x seq_len
        latent = v
        if use_esm:
            latent = v.transpose(1, 2)
        other = torch.repeat_interleave(latent, latent.shape[2], dim=0).T
        cos_sim = F.cosine_similarity(latent, other, dim=1).detach().numpy()
        print('layer', k, 'cosine', np.min(cos_sim), np.max(cos_sim), np.mean(cos_sim))
        # upper triangular matrix is cosine similarity
        cos_img = scale_to_color(cos_sim)
        cos_img = np.stack([np.triu(x, 0) for x in cos_img.T], axis=2)
        # save matrix as img
        
        if seq_img is not None:
            cos_img += seq_img
        cos_img = Image.fromarray(cos_img, mode='RGB')
        cos_img.save('{}{}.png'.format(filename, k))

        # separate image for attention weights
        if k > 0:
            # shape of attention is batch x layers x heads x seq_len x seq_len
            att = result['attentions'][0,k-1,:,:,:].detach().numpy()
            att_img = att_to_color(att)
            print('att', np.max(att_img[:,:,1]), np.max(att_img[:,:,2]), np.mean(att_img[:,:,1:3]))
            att_img = Image.fromarray(att_img, mode='HSV')
            att_img = att_img.convert(mode='RGB')
            # att_img = np.array(att_img)
            # [np.fill_diagonal(x, 0) for x in att_img.T]
            # if seq_img is not None:
            #     att_img += seq_img
            # att_img = Image.fromarray(att_img, mode='RGB')
            att_img.save('{}{}att.png'.format(filename, k))


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--deterministic', default=False, type=bool)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--max_epochs', default=1000, type=int)
    parser.add_argument('--gradient_clip_val', default=0.5, type=float)
    parser.add_argument('--val_check_interval', default=0, type=int)
    parser.add_argument('--limit_val_batches', default=0, type=int)
    parser.add_argument('--default_root_dir', default='.', type=str)
    parser.add_argument('--accumulate_grad_batches', default=1, type=int)
    parser.add_argument('--use_esm', default=False, type=bool)
    parser.add_argument('--use_pl', default=False, type=bool)
    parser.add_argument('--load_protein_model', default=False, type=bool)
    parser.add_argument('--n_class', default=9, type=int)
    parser.add_argument('--load_dump', default=False, type=bool)

    parser = PretrainESM.add_model_specific_args(parser)
    args = parser.parse_args()

    if args.load_protein_model:
        # Load ESM-1b model
        model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        batch_converter = alphabet.get_batch_converter()

        # Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
        data = [
            ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGGKALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
        ]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)

        # Extract per-residue representations (on CPU)
        result = model.forward(batch_tokens, repr_layers=[x for x in range(len(model.layers) + 1)],
                                need_head_weights=True, return_contacts=False)
        results_to_imgs(result, 'esm-1b-protein', use_esm=True)
    else:
        if args.use_esm:
            ModuleClass = PretrainESM
        else:
            ModuleClass = Pretrain

        module = ModuleClass(**vars(args))

        # generate test sequence from validation dataloader
        val_dl = module.val_dataloader()
        (seq, _, _), _ = next(iter(val_dl))
        seq = seq[0:1, :]  # first sequence only

        seq_img = seq_to_color(seq[0, :], base_map)
        # sequence on diagonal
        seq_img = np.stack([np.diag(x) for x in seq_img.T], axis=2)

        # for checkpoints in dir, load model
        for f in os.listdir(args.load_checkpoint_path):
            # run model on test sequence, save reprs at all layers
            ckpt_path = os.path.join(args.load_checkpoint_path, f)
            if args.load_dump:
                module.model = torch.load(ckpt_path, map_location=torch.device('cpu'))
                checkpoint = module
            elif args.use_pl:
                checkpoint = ModuleClass.load_from_checkpoint(ckpt_path, **vars(args))
            else:
                checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
            module.load_state_dict(checkpoint.state_dict())
            for k, v in module.state_dict().items():
                print(f, k, v.dtype, torch.min(v), torch.max(v), torch.mean(v))
            result = module.model.forward_with_intermediate_outputs(seq)

            results_to_imgs(result, f, seq_img=seq_img, use_esm=args.use_esm)
