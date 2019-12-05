import torch as th
import torch.nn as nn

import numpy as np

import torch_struct as ts

from latent_dialog.base_models import BaseModel
from latent_dialog.corpora import SYS, EOS, PAD
from latent_dialog.utils import INT, FLOAT, LONG, Pack
from latent_dialog.enc2dec.encoders import EncoderRNN, RnnUttEncoder, MlpGoalEncoder
from latent_dialog.nn_lib import IdentityConnector, Bi2UniConnector
from latent_dialog.enc2dec.decoders import DecoderRNN, GEN, GEN_VALID, TEACH_FORCE
from latent_dialog.criterions import NLLEntropy, NLLEntropy4CLF, CombinedNLLEntropy4CLF

import latent_dialog.utils as utils
import latent_dialog.nn_lib as nn_lib
import latent_dialog.criterions as criterions

#from torch_struct import gh

class Hmm(BaseModel):
    def __init__(self, corpus, config):
        super(Hmm, self).__init__(config)

        self.vocab = corpus.vocab
        self.vocab_dict = corpus.vocab_dict
        self.vocab_size = len(self.vocab)
        self.goal_vocab = corpus.goal_vocab
        self.goal_vocab_dict = corpus.goal_vocab_dict
        self.goal_vocab_size = len(self.goal_vocab)
        self.outcome_vocab = corpus.outcome_vocab
        self.outcome_vocab_dict = corpus.outcome_vocab_dict
        self.outcome_vocab_size = len(self.outcome_vocab)
        self.sys_id = self.vocab_dict[SYS]
        self.eos_id = self.vocab_dict[EOS]
        self.pad_id = self.vocab_dict[PAD]
        self.simple_posterior = False
        assert not config.simple_posterior

        self.goal_encoder = MlpGoalEncoder(goal_vocab_size=self.goal_vocab_size,
                                           k=config.k,
                                           nembed=config.goal_embed_size,
                                           nhid=config.goal_nhid,
                                           init_range=config.init_range)

        self.z_size = config.z_size
        self.item_emb = nn.Embedding(11, 32)
        self.res_layer = nn_lib.ResidualLayer(3 * 32, 64)
        self.w_pz0 = nn.Linear(64, 64, bias=False)
        self.prior_res_layer = nn_lib.ResidualLayer(config.ctx_cell_size, 64)

        self.embedding = nn.Embedding(self.vocab_size, config.embed_size, padding_idx=self.pad_id)
        self.utt_encoder = RnnUttEncoder(vocab_size=self.vocab_size,
                                         embedding_dim=config.embed_size,
                                         feat_size=0,
                                         goal_nhid=config.goal_nhid,
                                         rnn_cell=config.utt_rnn_cell,
                                         utt_cell_size=config.utt_cell_size,
                                         num_layers=config.num_layers,
                                         input_dropout_p=config.dropout,
                                         output_dropout_p=config.dropout,
                                         bidirectional=config.bi_utt_cell,
                                         variable_lengths=False, # means it looks at padding and 20 tokens every time
                                         use_attn=config.enc_use_attn,
                                         embedding=self.embedding)

        self.ctx_encoder = EncoderRNN(input_dropout_p=0.0,
                                      rnn_cell=config.ctx_rnn_cell,
                                      # input_size=self.utt_encoder.output_size+config.goal_nhid,
                                      input_size=self.utt_encoder.output_size,
                                      hidden_size=config.ctx_cell_size,
                                      num_layers=config.num_layers,
                                      output_dropout_p=config.dropout,
                                      bidirectional=config.bi_ctx_cell,
                                      variable_lengths=False)
        # mu and logvar projector
        self.c2z = nn_lib.Hidden2DiscreteDeal(
            self.ctx_encoder.output_size, config.z_size,
            is_lstm=config.ctx_rnn_cell == 'lstm',
        )

        self.xc2z = nn_lib.Hidden2DiscreteDeal(
            self.ctx_encoder.output_size + self.utt_encoder.output_size,
            config.z_size, is_lstm=False,
        )

        self.gumbel_connector = nn_lib.GumbelConnector(config.use_gpu)
        #self.z_embedding = nn.Linear(config.z_size, config.dec_cell_size, bias=False)
        self.z_embedding = nn.Embedding(config.z_size, config.dec_cell_size)

        self.decoder = DecoderRNN(input_dropout_p=config.dropout,
                                  rnn_cell=config.dec_rnn_cell,
                                  input_size=config.embed_size + config.goal_nhid,
                                  hidden_size=config.dec_cell_size,
                                  num_layers=config.num_layers,
                                  output_dropout_p=config.dropout,
                                  bidirectional=False,
                                  vocab_size=self.vocab_size,
                                  use_attn=config.dec_use_attn,
                                  ctx_cell_size=self.ctx_encoder.output_size,
                                  attn_mode=config.dec_attn_mode,
                                  sys_id=self.sys_id,
                                  eos_id=self.eos_id,
                                  use_gpu=config.use_gpu,
                                  max_dec_len=config.max_dec_len,
                                  embedding=self.embedding)

        self.nll = NLLEntropy(self.pad_id, config.avg_type)
        self.cat_kl_loss = criterions.CatKLLoss()
        self.entropy_loss = criterions.Entropy()

        # ?
        self.log_uniform_z = th.log(th.ones(1) / config.z_size)
        if self.use_gpu:
            self.log_uniform_z = self.log_uniform_z.cuda()

    def valid_loss(self, loss, batch_cnt=None):
        return loss.nll

    # for decoding? 
    def z2dec(self, last_h, requires_grad):
        import pdb; pdb.set_trace()
        logits, log_qy = self.c2z(last_h)

        if requires_grad:
            sample_y = self.gumbel_connector(logits)
            logprob_z = None
        else:
            idx = th.multinomial(th.exp(log_qy), 1).detach()
            logprob_z = th.sum(log_qy.gather(1, idx))
            sample_y = utils.cast_type(Variable(th.zeros(log_qy.size())), FLOAT, self.use_gpu)
            sample_y.scatter_(1, idx, 1.0)

        if self.config.dec_use_attn:
            z_embeddings = th.t(self.z_embedding.weight).split(self.config.k_size, dim=0)
            attn_context = []
            temp_sample_y = sample_y.view(-1, self.config.y_size, self.config.k_size)
            for z_id in range(self.config.y_size):
                attn_context.append(th.mm(temp_sample_y[:, z_id], z_embeddings[z_id]).unsqueeze(1))
            attn_context = th.cat(attn_context, dim=1)
            dec_init_state = th.sum(attn_context, dim=1).unsqueeze(0)
        else:
            attn_context = None
            dec_init_state = self.z_embedding(sample_y.view(1, -1, self.config.y_size * self.config.k_size))

        return dec_init_state, attn_context, logprob_z

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
            transition_matrix = transition_matrix.masked_fill(mask.unsqueeze(-2), float("-inf"))
        #return pz0.log_softmax(-1), transition_matrix.log_softmax(-1).transpose(-2, -1)
        #return pz0, transition_matrix.transpose(-2, -1)
        return phi_pzt, transition_matrix

    def forward(self, data_feed, mode, clf=False, gen_type='greedy', use_pz=None, return_latent=False):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        ctx_utts = self.np2var(data_feed['contexts'], LONG)  # (batch_size, max_ctx_len, max_utt_len)
        out_utts = self.np2var(data_feed['outputs'], LONG)  # (batch_size, max_out_len)
        goals = self.np2var(data_feed['goals'], LONG)  # (batch_size, goal_len)
        partitions = self.np2var(data_feed.partitions, LONG)
        num_partitions = self.np2var(data_feed.num_partitions, INT)
        # effective batch size
        batch_size = len(ctx_lens)
        true_batch_size = data_feed.dlg_idxs.max()

        # encode goal info
        goals_h = self.goal_encoder(goals)  # (batch_size, goal_nhid)

        enc_inputs, _, _ = self.utt_encoder(ctx_utts, goals=goals_h)
        # (batch_size, max_ctx_len, num_directions*utt_cell_size)

        # : (batch_size, max_ctx_len, ctx_cell_size)
        # enc_last: tuple, (h_n, c_n)
        # h_n: (num_layers*num_directions, batch_size, ctx_cell_size)
        # c_n: (num_layers*num_directions, batch_size, ctx_cell_size)
        enc_outs, enc_last = self.ctx_encoder(enc_inputs, input_lengths=ctx_lens, goals=None)

        # get decoder inputs
        dec_inputs = out_utts[:, :-1]
        labels = out_utts[:, 1:].contiguous()

        # TODO: make this directed? then it's an MEMM
        #logits_pzt, log_pzt = self.c2z(enc_last)

        # encode response and use posterior to find q(z|x, c)
        #x_h, _, _ = self.utt_encoder(out_utts.unsqueeze(1), goals=goals_h)
        #logits_qz_t, log_qz_t = self.xc2z(th.cat([enc_last, x_h.squeeze(1).unsqueeze(0)], dim=2))

        ctx_input = self.prior_res_layer(enc_last.squeeze(0))
        state_emb = self.res_layer(self.item_emb(partitions).view(-1, self.z_size, 3 * 32))
        # REMINDER: psi_zr_zl = psi(z_t, z_t-1) (z right, z left)
        #_, psi_zr_zl= self.hmm_potentials(state_emb, lengths=num_partitions, dlg_lens=data_feed.dlg_lens)
        phi_zt, psi_zl_zr = self.hmm_potentials(state_emb, ctx_input, lengths=num_partitions)
        logp_zt,  logp_zr_zl = phi_zt.log_softmax(-1), psi_zl_zr.log_softmax(-1).transpose(-1, -2)

        # repeat EVERYTHING, add state_embs to `goals_h` and get scores?
        # TODO: don't hijack, add a new input so we can attend to it

        # decode
        N, T = dec_inputs.shape
        N, H = goals_h.shape
        dec_init_state = enc_last.repeat(1, self.z_size, 1)
        dec_outputs, dec_hidden_state, ret_dict = self.decoder(
            batch_size = batch_size,
            dec_inputs = dec_inputs.repeat(1, self.z_size, 1).view(-1, T), # (batch_size, response_size-1)
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
        # TODO: are we going to run into numerical stability issues?
        # might need to pretrain to assign more mass to correct sentences.
        dlg_idxs = data_feed.dlg_idxs
        prev_zt = logp_zt[0]
        log_pxt = [
            (logp_xt_zt[0] + prev_zt).logsumexp(-1)
        ]
        for t in range(1, N):
            if dlg_idxs[t] != dlg_idxs[t-1]:
                # restart hmm
                prev_zt = phi_zt[t]
                log_pxt.append(
                    (logp_xt_zt[t] + prev_zt).logsumexp(-1)
                )
            else:
                # continue
                prev_zt = (prev_zt.unsqueeze(-2) + logp_zr_zl[t]).logsumexp(-1)
                log_pxt.append(
                    (logp_xt_zt[t] + prev_zt).logsumexp(-1)
                )

        nll = -sum(log_pxt)
        if self.nll.avg_type == "seq":
            nll = nll / N
        elif self.nll.avg_type == "real_word":
            nll = nll / (labels == self.nll.padding_idx).sum()
        else:
            raise ValueError("Unknown average type")

        if mode == GEN:
            return ret_dict, labels
        else:
            results = Pack(nll=nll)

            if return_latent:
                results['latent_action'] = dec_init_state

            return results