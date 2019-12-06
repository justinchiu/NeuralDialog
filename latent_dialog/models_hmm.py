import torch as th
import torch.nn as nn
from torch.autograd import Variable
from latent_dialog.base_models import BaseModel
from latent_dialog.models_deal import HRED
from latent_dialog.corpora import SYS, EOS, PAD
from latent_dialog.utils import INT, FLOAT, LONG, Pack
from latent_dialog.enc2dec.encoders import EncoderRNN, RnnUttEncoder, MlpGoalEncoder
from latent_dialog.nn_lib import IdentityConnector, Bi2UniConnector
from latent_dialog.enc2dec.decoders import DecoderRNN, GEN, GEN_VALID, TEACH_FORCE
from latent_dialog.criterions import NLLEntropy, NLLEntropy4CLF, CombinedNLLEntropy4CLF
import latent_dialog.utils as utils
import latent_dialog.nn_lib as nn_lib
import latent_dialog.criterions as criterions
import numpy as np


class Hmm(HRED):
    def __init__(self, corpus, config):
        super(Hmm, self).__init__(corpus, config)

        self.z_embedding = nn.Embedding(config.z_size, config.dec_cell_size)
        self.z_size = config.z_size
        self.book_emb = nn.Embedding(11, 32)
        self.hat_emb = nn.Embedding(11, 32)
        self.ball_emb = nn.Embedding(11, 32)
        self.res_layer = nn_lib.ResidualLayer(3 * 32, 64)
        self.w_pz0 = nn.Linear(64, 64, bias=False)
        self.prior_res_layer = nn_lib.ResidualLayer(config.ctx_cell_size, 64)

    def hmm_potentials(self, emb, ctx, lengths=None, dlg_lens=None):
        # (a) - [a,b] - (b), then transpose potentials later for torch_struct
        phi_pzt = th.einsum("nzh,nh->nz", emb, ctx)
        transition_matrix = th.einsum("nac,nbc->nab", emb + ctx.unsqueeze(1), emb)
        if lengths is not None:
            N = lengths.shape[0]
            mask = ~(
                th.arange(self.z_size, device = lengths.device, dtype = lengths.dtype)
                    .repeat(N, 1) < lengths.unsqueeze(-1)
            )
            phi_pzt = phi_pzt.masked_fill(mask, float("-inf"))
            #transition_matrix1 = transition_matrix.masked_fill(mask.unsqueeze(-2), float("-inf"))
            transition_matrix = transition_matrix.masked_fill(mask.unsqueeze(-2), -1e12)
        #return pz0.log_softmax(-1), transition_matrix.log_softmax(-1).transpose(-2, -1)
        #return pz0, transition_matrix.transpose(-2, -1)
        return phi_pzt, transition_matrix

    def forward(self, data_feed, mode, clf=False, gen_type='greedy', use_py=None, return_latent=False):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        ctx_utts = self.np2var(data_feed['contexts'], LONG)  # (batch_size, max_ctx_len, max_utt_len)
        ctx_confs = self.np2var(data_feed['context_confs'], FLOAT)  # (batch_size, max_ctx_len)
        out_utts = self.np2var(data_feed['outputs'], LONG)  # (batch_size, max_out_len)
        goals = self.np2var(data_feed['goals'], LONG)  # (batch_size, goal_len)
        partitions = self.np2var(data_feed.partitions, LONG)
        num_partitions = self.np2var(data_feed.num_partitions, INT)
        batch_size = len(ctx_lens)

        # encode goal info
        goals_h = self.goal_encoder(goals)  # (batch_size, goal_nhid)

        enc_inputs, _, _ = self.utt_encoder(
            ctx_utts,
            feats=ctx_confs,
            goals=goals_h,  # (batch_size, max_ctx_len, num_directions*utt_cell_size)
        )

        # enc_outs: (batch_size, max_ctx_len, ctx_cell_size)
        # enc_last: tuple, (h_n, c_n)
        # h_n: (num_layers*num_directions, batch_size, ctx_cell_size)
        # c_n: (num_layers*num_directions, batch_size, ctx_cell_size)
        enc_outs, enc_last = self.ctx_encoder(enc_inputs, input_lengths=ctx_lens, goals=None)

        # get decoder inputs
        dec_inputs = out_utts[:, :-1]
        labels = out_utts[:, 1:].contiguous()

        # pack attention context
        if self.config.dec_use_attn:
            attn_context = enc_outs
        else:
            attn_context = None

        # create decoder initial states
        dec_init_state = self.connector(enc_last)

		# attend over enc_outs
        ctx_input = self.prior_res_layer(enc_last.squeeze(0))
        state_emb = self.res_layer(th.cat([
            self.book_emb(partitions[:,:,0]),
            self.hat_emb (partitions[:,:,1]),
            self.ball_emb(partitions[:,:,2]),
        ], -1))

        phi_zt, psi_zl_zr = self.hmm_potentials(state_emb, ctx_input, lengths=num_partitions)
        logp_zt = phi_zt.log_softmax(-1)
        logp_zr_zl = psi_zl_zr.log_softmax(-1).transpose(-1, -2)

        # decode
        N, T = dec_inputs.shape
        N, H = goals_h.shape
        dec_init_state = enc_last.repeat(1, self.z_size, 1)
        dec_outputs, dec_hidden_state, ret_dict = self.decoder(
            batch_size = batch_size,
            dec_inputs = dec_inputs.repeat(1, self.z_size, 1).view(-1, T), # (batch_size, response_s            dec_inputs = dec_inputs, # (batch_size, response_size-1)
            dec_init_state = dec_init_state,  # tuple: (h, c)
            attn_context = None, # (batch_size, max_ctx_len, ctx_cell_size)
            mode = mode,
            gen_type = gen_type,
            beam_size = self.config.beam_size,
            goal_hid = (goals_h.unsqueeze(1) + state_emb).view(-1, H),  # (batch_size, goal_nhid)
        )

        BLAM, T, V = dec_outputs.shape
        # all word probs, they need to be summed over
        # `log p(xt) = \sum_i \log p(w_ti)`
        logp_wt_zt = dec_outputs.view(N, self.z_size, T, V).gather(
            -1,
            labels.view(N, 1, T, 1).expand(N, self.z_size, T, 1),
        ).squeeze(-1)

        # get rid of padding, mask to 0
        logp_xt_zt = (logp_wt_zt
            .masked_fill(labels.unsqueeze(1) == self.nll.padding_idx, 0)
            .sum(-1)
        )

        # do linear chain stuff
        # a little weird, we're working with a chain graphical model
        # need to normalize over each zt so the lm probs remain normalized
        dlg_idxs = data_feed.dlg_idxs
        prev_zt = logp_zt[0]
        logp_xt = [
            (logp_xt_zt[0] + prev_zt).logsumexp(-1)
        ]
        for t in range(1, N):
            if dlg_idxs[t] != dlg_idxs[t-1]:
                # restart hmm
                prev_zt = logp_zt[t]
                logp_xt.append(
                    (logp_xt_zt[t] + prev_zt).logsumexp(-1)
                )
            else:
                # continue
                # unsqueeze is unnecessary, broadcasting handles it
                meh = prev_zt
                prev_zt = (prev_zt.unsqueeze(-2) + logp_zr_zl[t]).logsumexp(-1)
                logp_xt.append(
                    (logp_xt_zt[t] + prev_zt).logsumexp(-1)
                )
        logp_xt = th.stack(logp_xt)
        if self.nll.avg_type == "real_word":
            nll = -(logp_xt / (labels.sign().sum(-1).float())).mean()
        elif self.nll.avg_type == "word":
            nll = -(logp_xt.sum() / labels.sign().sum())
        else:
            raise ValueError("Unknown reduction type")

        if mode == GEN:
            return ret_dict, labels
        if return_latent:
            return Pack(nll=nll,
                        latent_action=dec_init_state)
        else:
            return Pack(nll=nll)


    # for decoding when negotiating
    def z2dec(self, last_h, requires_grad):
        import pdb; pdb.set_trace()
        raise NotImplementedError("not adapted yet")
