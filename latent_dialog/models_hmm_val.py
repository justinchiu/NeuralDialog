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


class HmmVal(BaseModel):
    def __init__(self, corpus, config):
        super(HmmVal, self).__init__(config)

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

        self.z_embedding = nn.Embedding(config.z_size, config.dec_cell_size)
        self.z_size = config.z_size
        self.book_emb = nn.Embedding(11, 32)
        self.hat_emb = nn.Embedding(11, 32)
        self.ball_emb = nn.Embedding(11, 32)
        self.res_layer = nn_lib.ResidualLayer(3 * 32, 64)
        self.w_pz0 = nn.Linear(64, 64, bias=False)
        self.prior_res_layer = nn_lib.ResidualLayer(config.ctx_cell_size, 64)

    def hmm_potentials(self, emb, ctx, lengths=None, dlg_lens=None, size=128):
        # (a) - [a,b] - (b)
        phi_pzt = th.einsum("nzh,nh->nz", emb, ctx)
        transition_matrix = th.einsum("nac,nbc->nab", emb + ctx.unsqueeze(1), emb)
        if lengths is not None:
            N = lengths.shape[0]
            mask = ~(
                th.arange(size, device = lengths.device, dtype = lengths.dtype)
                    .repeat(N, 1) < lengths.unsqueeze(-1)
            )
            phi_pzt = phi_pzt.masked_fill(mask, float("-inf"))
            transition_matrix = transition_matrix.masked_fill(mask.unsqueeze(-2), -1e12)
        return phi_pzt, transition_matrix

    def forward(
        self, data_feed, mode, clf=False, gen_type='greedy',
        use_py=None, return_latent=False,
    ):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        ctx_utts = self.np2var(data_feed['contexts'], LONG)  # (batch_size, max_ctx_len, max_utt_len)
        ctx_confs = self.np2var(data_feed['context_confs'], FLOAT)  # (batch_size, max_ctx_len)
        out_utts = self.np2var(data_feed['outputs'], LONG)  # (batch_size, max_out_len)
        goals = self.np2var(data_feed['goals'], LONG)  # (batch_size, goal_len)
        partitions = self.np2var(data_feed.partitions, LONG)
        num_partitions = self.np2var(data_feed.num_partitions, INT)
        batch_size = len(ctx_lens)

        partner_goals = self.np2var(data_feed.partner_goals, LONG)
        num_partner_goals = self.np2var(data_feed.num_partner_goals, INT)

        N, G, _ = partner_goals.shape
        p_goals_h = self.goal_encoder(partner_goals.view(-1, 6)).view(N, G, -1)

        size = G

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

        #import pdb; pdb.set_trace()
        #phi_zt, psi_zl_zr = self.hmm_potentials(state_emb, ctx_input, lengths=num_partitions)
        phi_zt, psi_zl_zr = self.hmm_potentials(
            p_goals_h, ctx_input,
            lengths = num_partner_goals,
            size = G,
        )
        logp_zt = phi_zt.log_softmax(-1)
        logp_zr_zl = psi_zl_zr.log_softmax(-1).transpose(-1, -2)

        # decode
        N, T = dec_inputs.shape
        N, H = goals_h.shape
        dec_init_state = enc_last.repeat(1, size, 1)
        dec_outputs, dec_hidden_state, ret_dict = self.decoder(
            batch_size = batch_size,
            dec_inputs = dec_inputs.repeat(1, size, 1).view(-1, T), # (batch_size, response_size-1)
            dec_init_state = dec_init_state,  # tuple: (h, c)
            attn_context = None, # (batch_size, max_ctx_len, ctx_cell_size)
            mode = mode,
            gen_type = gen_type,
            beam_size = self.config.beam_size,
            goal_hid = th.cat([
                goals_h.unsqueeze(1).expand_as(p_goals_h),
                p_goals_h,
            ], -1).view(-1, 2*H),  # (batch_size, goal_nhid)
        )

        BLAM, T, V = dec_outputs.shape
        # all word probs, they need to be summed over
        # `log p(xt) = \sum_i \log p(w_ti)`
        logp_wt_zt = dec_outputs.view(N, size, T, V).gather(
            -1,
            labels.view(N, 1, T, 1).expand(N, size, T, 1),
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
        raise NotImplementedError("not adapted yet")
