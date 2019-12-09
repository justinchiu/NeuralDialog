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


class HRED(BaseModel):
    def __init__(self, corpus, config):
        super(HRED, self).__init__(config)

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
                                  input_size=config.embed_size + 2*config.goal_nhid,
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

        # oracle modules
        self.book_emb = nn.Embedding(16, 32)
        self.hat_emb = nn.Embedding(16, 32)
        self.ball_emb = nn.Embedding(16, 32)
        self.res_layer = nn_lib.ResidualLayer(3 * 32, 128)

        self.book_emb_out = nn.Embedding(16, 32)
        self.hat_emb_out = nn.Embedding(16, 32)
        self.ball_emb_out = nn.Embedding(16, 32)
        self.res_layer_out = nn_lib.ResidualLayer(3 * 32, 128)

        self.prop_utt_encoder = RnnUttEncoder(
            vocab_size=self.vocab_size,
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
            embedding=self.embedding,
        )

        self.prop_ctx_encoder = EncoderRNN(
            input_dropout_p=0.0,
            rnn_cell=config.ctx_rnn_cell,
            # input_size=self.utt_encoder.output_size+config.goal_nhid, 
            input_size = self.utt_encoder.output_size + 64
            if config.oracle_context else self.utt_encoder.output_size,
            hidden_size=config.ctx_cell_size,
            num_layers=config.num_layers,
            output_dropout_p=config.dropout,
            bidirectional=config.bi_ctx_cell,
            variable_lengths=False,
        )


        self.w_pz0 = nn.Linear(64, 64, bias=False)
        self.prior_res_layer = nn_lib.ResidualLayer(config.ctx_cell_size, 64)
        self.res_goal_mlp = nn_lib.ResidualLayer(256 + config.goal_nhid, 128)

    def forward(self, data_feed, mode, clf=False, gen_type='greedy', use_py=None, return_latent=False, get_marginals=False):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        ctx_utts = self.np2var(data_feed['contexts'], LONG)  # (batch_size, max_ctx_len, max_utt_len)
        ctx_confs = self.np2var(data_feed['context_confs'], FLOAT)  # (batch_size, max_ctx_len)
        out_utts = self.np2var(data_feed['outputs'], LONG)  # (batch_size, max_out_len)
        goals = self.np2var(data_feed['goals'], LONG)  # (batch_size, goal_len)
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

        partitions = self.np2var(data_feed.partitions, LONG)
        num_partitions = self.np2var(data_feed.num_partitions, INT)
        # oracle input
        partner_goals = self.np2var(data_feed.true_partner_goals, LONG)
        parsed_outputs = self.np2var(data_feed.parsed_outputs, LONG)
        # true partner item values
        partner_goals_h = self.goal_encoder(partner_goals)

        # proposal prediction
        prop_enc_inputs, _, _ = self.prop_utt_encoder(
            ctx_utts,
            feats=ctx_confs,
            goals=goals_h,  # (batch_size, max_ctx_len, num_directions*utt_cell_size)
        )

        # enc_outs: (batch_size, max_ctx_len, ctx_cell_size)
        # enc_last: tuple, (h_n, c_n)
        # h_n: (num_layers*num_directions, batch_size, ctx_cell_size)
        # c_n: (num_layers*num_directions, batch_size, ctx_cell_size)
        prop_enc_outs, prop_enc_last = self.prop_ctx_encoder(
            enc_inputs if self.config.tie_prop_utt_enc else prop_enc_inputs,
            input_lengths = ctx_lens,
            goals = partner_goals_h if self.config.oracle_context else None,
        )

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

        big_goals_h = self.res_goal_mlp(th.cat([
            goals_h.unsqueeze(1).expand(
                state_emb_out.shape[0],
                state_emb_out.shape[1],
                goals_h.shape[-1],
            ),
            state_emb_out,
        ], -1))

        z_size = partitions.shape[1]

        prop_mask = (partitions == parsed_outputs.unsqueeze(1)).all(-1)
        logits_prop = th.einsum("nsh,nh->ns", state_emb_out, prop_enc_last[-1])
        mask = ~(
            th.arange(z_size, device = num_partitions.device, dtype = num_partitions.dtype)
                .repeat(partitions.shape[0], 1) < num_partitions.unsqueeze(-1)
        )
        logp_prop = logits_prop.masked_fill(mask, float("-inf")).log_softmax(-1) # get decoder inputs

        if self.config.semisupervised:
            # re-use params or make new ones? re-using can only hurt
            # TODO: use new parameters
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
            noise_state_emb = th.cat([my_state_emb, your_state_emb], -1)
            logp_tprop_prop = th.einsum("nth,nsh->nts", noise_state_emb, state_emb_out).log_softmax(1)
            nll_prop = - self.config.prop_weight * (logp_tprop_prop + logp_prop.unsqueeze(-2)).logsumexp(-1)[prop_mask].mean()
        else:
            nll_prop = - self.config.prop_weight * logp_prop[prop_mask].mean()

        dec_inputs = out_utts[:, :-1]
        labels = out_utts[:, 1:].contiguous()

        # pack attention context
        if self.config.dec_use_attn:
            attn_context = enc_outs
        else:
            attn_context = None

        # create decoder initial states
        dec_init_state = self.connector(enc_last)

        if mode == GEN:
            N, Z, H = big_goals_h.shape
            if gen_type == "sampled":
                sampled_proposal_indices = logp_prop.exp().multinomial(1)
            elif gen_type == "greedy":
                sampled_proposal_indices = logp_prop.argmax(-1)
            else:
                raise ValueError(f"Unknown gen_type: {gen_type}")
            sampled_goals_h = big_goals_h.gather(
                1, sampled_proposal_indices.view(N, 1, 1).expand(N, 1, H)).squeeze(1)
            dec_outputs, dec_hidden_state, ret_dict = self.decoder(
                batch_size = batch_size,
                dec_inputs = dec_inputs,
                # (batch_size, response_size-1)
                dec_init_state = dec_init_state,
                attn_context = attn_context,
                # (batch_size, max_ctx_len, ctx_cell_size)
                mode = mode,
                gen_type = gen_type,
                beam_size=self.config.beam_size,
                # my goal, your goal, and the proposal!!! a lot
                goal_hid = sampled_goals_h,
                #goal_hid=big_goals_h[prop_mask],
            )  # (batch_size, goal_nhid)
            return ret_dict, labels

        # decode
        N, T = dec_inputs.shape
        H = dec_init_state.shape[-1]
        dec_outputs, dec_hidden_state, ret_dict = self.decoder(
            batch_size = batch_size * z_size,
            dec_inputs = dec_inputs.repeat(1, z_size).view(-1, T),
            # (batch_size, response_size-1)
            dec_init_state = dec_init_state.repeat(1, 1, z_size).view(1, z_size * batch_size, H),
            attn_context = attn_context,
            # (batch_size, max_ctx_len, ctx_cell_size)
            mode = mode,
            gen_type = gen_type,
            beam_size=self.config.beam_size,
            # my goal, your goal, and the proposal!!! a lot
            goal_hid=big_goals_h.view(-1, 128),
        )  # (batch_size, goal_nhid)
        V = dec_outputs.shape[-1]
        #logp_w_prop = dec_outputs.view(N, z_size, T, V)[prop_mask]
        T_out = dec_outputs.shape[-2]
        logp_w_prop = (
            dec_outputs.view(N, z_size, T_out, V) + logp_prop.view(N, z_size, 1, 1)
        ).logsumexp(1)

        if get_marginals:
            return Pack(
                dec_outputs = dec_outputs,
                labels = labels,
            )
        if mode == GEN:
            return ret_dict, labels
        if return_latent:
            return Pack(nll=self.nll(logp_w_prop, labels),
                        latent_action=dec_init_state)
        else:
            return Pack(nll=self.nll(logp_w_prop, labels), nll_prop = nll_prop)


    def z2dec(self, state_emb_out, prop_enc_last):
        logits_prop = th.einsum("nsh,nh->ns", state_emb_out, prop_enc_last[-1])
        logp_prop = logits_prop.log_softmax(-1)
        prop_index  = logits_prop.exp().multinomial(1)
        return prop_index, logp_prop.gather(1, prop_index)
