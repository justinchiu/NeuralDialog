import torch as th
import torch.nn as nn
from torch.autograd import Variable
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
import numpy as np


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

        self.goal_encoder = MlpGoalEncoder(goal_vocab_size=self.goal_vocab_size,
                                           k=config.k,
                                           nembed=config.goal_embed_size,
                                           nhid=config.goal_nhid,
                                           init_range=config.init_range)

        self.embedding = nn.Embedding(self.vocab_size, config.embed_size, padding_idx=self.pad_id)
        self.utt_encoder = RnnUttEncoder(vocab_size=self.vocab_size,
                                         embedding_dim=config.embed_size,
                                         feat_size=1,
                                         goal_nhid=config.goal_nhid,
                                         rnn_cell=config.utt_rnn_cell,
                                         utt_cell_size=config.utt_cell_size,
                                         num_layers=config.num_layers,
                                         input_dropout_p=config.dropout,
                                         output_dropout_p=config.dropout,
                                         bidirectional=config.bi_utt_cell,
                                         variable_lengths=False,
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

        # TODO connector
        if config.bi_ctx_cell:
            self.connector = Bi2UniConnector(rnn_cell=config.ctx_rnn_cell,
                                             num_layer=1,
                                             hidden_size=config.ctx_cell_size,
                                             output_size=config.dec_cell_size)
        else:
            self.connector = IdentityConnector()

        self.decoder = DecoderRNN(input_dropout_p=config.dropout,
                                  rnn_cell=config.dec_rnn_cell,
                                  input_size=config.embed_size + config.goal_nhid + 64,
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


        # new hmm stuff
        self.noisy_proposal_labels = config.noisy_proposal_labels
         
        self.z_size = config.z_size

        # for the transition matrix
        self.book_emb = nn.Embedding(16, 32)
        self.hat_emb = nn.Embedding(16, 32)
        self.ball_emb = nn.Embedding(16, 32)
        self.res_layer = nn_lib.ResidualLayer(3 * 32, 64)

        self.book_emb_out = nn.Embedding(16, 32)
        self.hat_emb_out = nn.Embedding(16, 32)
        self.ball_emb_out = nn.Embedding(16, 32)
        self.res_layer_out = nn_lib.ResidualLayer(3 * 32, 64)

        self.res_goal_mlp = nn_lib.ResidualLayer(64 * 3, 64 * 2)

        self.w_pz0 = nn.Linear(64, 64, bias=False)
        self.prior_res_layer = nn_lib.ResidualLayer(config.ctx_cell_size, 2*64)


    def hmm_potentials(self, emb, emb_out, ctx, lengths=None, dlg_lens=None):
        # (a) - [a,b] - (b)
        phi_pzt = th.einsum("nzh,nh->nz", emb, ctx)
        transition_matrix = th.einsum("nac,nbc->nab", emb + ctx.unsqueeze(1), emb_out)
        if lengths is not None:
            N = lengths.shape[0]
            mask = ~(
                th.arange(self.z_size, device = lengths.device, dtype = lengths.dtype)
                    .repeat(N, 1) < lengths.unsqueeze(-1)
            )
            phi_pzt = phi_pzt.masked_fill(mask, float("-inf"))
            transition_matrix = transition_matrix.masked_fill(mask.unsqueeze(-2), -1e12)
        return phi_pzt, transition_matrix

    def forward(
        self, data_feed, mode, clf=False, gen_type='greedy',
        use_py=None, return_latent=False,
        get_marginals = False,
    ):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        ctx_utts = self.np2var(data_feed['contexts'], LONG)  # (batch_size, max_ctx_len, max_utt_len)
        ctx_confs = self.np2var(data_feed['context_confs'], FLOAT)  # (batch_size, max_ctx_len)
        out_utts = self.np2var(data_feed['outputs'], LONG)  # (batch_size, max_out_len)
        goals = self.np2var(data_feed['goals'], LONG)  # (batch_size, goal_len)
        partitions = self.np2var(data_feed.partitions, LONG)
        num_partitions = self.np2var(data_feed.num_partitions, INT)
        batch_size = len(ctx_lens)

        self.z_size = data_feed.num_partitions.max()

        # oracle
        parsed_outputs = self.np2var(data_feed.parsed_outputs, LONG)
        partner_goals = self.np2var(data_feed.true_partner_goals, LONG)

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

        # transition matrix
        ctx_input = self.prior_res_layer(enc_last[-1])
        my_state_emb = self.res_layer(th.cat([
            self.book_emb(partitions[:,:,0]),
            self.hat_emb (partitions[:,:,1]),
            self.ball_emb(partitions[:,:,2]),
        ], -1))
        your_state_emb = self.res_layer(th.cat([
            self.book_emb(partitions[:,:,3]),
            self.hat_emb (partitions[:,:,4]),
            self.ball_emb(partitions[:,:,5]),
        ], -1))
        state_emb = th.cat([my_state_emb, your_state_emb], -1)
        my_state_emb_out = self.res_layer_out(th.cat([
            self.book_emb_out(partitions[:,:,0]),
            self.hat_emb_out (partitions[:,:,1]),
            self.ball_emb_out(partitions[:,:,2]),
        ], -1))
        your_state_emb_out = self.res_layer_out(th.cat([
            self.book_emb_out(partitions[:,:,3]),
            self.hat_emb_out (partitions[:,:,4]),
            self.ball_emb_out(partitions[:,:,5]),
        ], -1))
        state_emb_out = th.cat([my_state_emb_out, your_state_emb_out], -1)

        goals_h = self.res_goal_mlp(th.cat([
            goals_h.unsqueeze(1).expand(state_emb.shape[0], state_emb.shape[1], goals_h.shape[-1]),
            state_emb,
        ], -1)).view(-1, 128)

        # for noisy labels
        if self.noisy_proposal_labels:
            # transition from state to label
            label_mask = (partitions == parsed_outputs.unsqueeze(1)).all(-1)
            logp_label_z = th.einsum("nsh,nth->nts", state_emb, state_emb_out).log_softmax(-1)
            # outer dim t should be output label

        phi_zt, psi_zl_zr = self.hmm_potentials(state_emb, state_emb_out, ctx_input, lengths=num_partitions)
        logp_zt = phi_zt.log_softmax(-1)
        logp_zr_zl = psi_zl_zr.log_softmax(-1).transpose(-1, -2)

        # decode
        N, T = dec_inputs.shape
        dec_init_state = enc_last.repeat(1, 1, self.z_size).view(self.config.num_layers, N*self.z_size, -1)
        dec_outputs, dec_hidden_state, ret_dict = self.decoder(
            batch_size = batch_size * self.z_size,
            dec_inputs = dec_inputs.repeat(1, 1, self.z_size).view(-1, T), # (batch_size, response_size-1)
            dec_init_state = dec_init_state,  # tuple: (h, c)
            attn_context = None, # (batch_size, max_ctx_len, ctx_cell_size)
            mode = mode,
            gen_type = gen_type,
            beam_size = self.config.beam_size,
            goal_hid = goals_h,  # (batch_size, goal_nhid)
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
        t = 0
        ll_label = 0
        prev_zt = logp_zt[t]

        logp_xt = [
            (logp_xt_zt[t] + prev_zt).logsumexp(-1)
        ]
        if self.training and self.noisy_proposal_labels and label_mask[0].any():
            if not self.config.sup_proposal_labels:
                # predict noisy proposal from hidden state
                ll_label += (
                    logp_label_z[t] + prev_zt.unsqueeze(-1)
                ).logsumexp(0)[label_mask[t]].logsumexp(0)
            else:
                ll_label += prev_zt[label_mask[t]].logsumexp(0)
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
                #prev_zt = (prev_zt.unsqueeze(-2) + logp_zr_zl[t]).logsumexp(-1)
                prev_zt = logp_zt[t]
                logp_xt.append(
                    (logp_xt_zt[t] + prev_zt).logsumexp(-1)
                )
            if self.training and self.noisy_proposal_labels and label_mask[t].any():
                if not self.config.sup_proposal_labels:
                    # predict noisy proposal from hidden state
                    ll_label += (
                        logp_label_z[t] + prev_zt.unsqueeze(-1)
                    ).logsumexp(0)[label_mask[t]].logsumexp(0)
                else:
                    ll_label += prev_zt[label_mask[t]].logsumexp(0)
        logp_xt = th.stack(logp_xt)
        if self.nll.avg_type == "real_word":
            nll_word = -(logp_xt / (labels.sign().sum(-1).float())).mean()
        elif self.nll.avg_type == "word":
            nll_word = -(logp_xt.sum() / labels.sign().sum())
        else:
            raise ValueError("Unknown reduction type")

        if self.training and self.noisy_proposal_labels and label_mask.any():
            #nll -= 0.1 * ll_label / label_mask.sum().float()
            nll_label = - self.config.label_weight * ll_label / label_mask.any(-1).sum().float()
        else:
            nll_label = th.zeros(1).to(nll_word.device)

            #import pdb; pdb.set_trace()
        if get_marginals:
            return Pack(
                dec_outputs = dec_outputs,
                logp_xt = logp_xt,
                labels = labels,
            )
        #Z = prev_zt.logsumexp(0)
        if mode == GEN:
            return ret_dict, labels
        if return_latent:
            return Pack(nll=nll,
                        latent_action=dec_init_state)
        else:
            return Pack(nll_label=nll_label, nll_word=nll_word)


    # for decoding when negotiating
    def z2dec(self, last_h, requires_grad):
        raise NotImplementedError("not adapted yet")
