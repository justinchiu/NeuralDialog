import numpy as np
from latent_dialog.utils import Pack
from latent_dialog.base_data_loaders import BaseDataLoaders
from latent_dialog.corpora import USR, SYS
from latent_dialog.contexts import get_valid_contexts_ints

from latent_dialog.latents import get_latent_powerset

import json


class DealDataLoaders(BaseDataLoaders):
    def __init__(self, name, data, config):
        super(DealDataLoaders, self).__init__(name)
        self.max_utt_len = config.max_utt_len
        self.seq = config.seq
        self.data = (
            self.flatten_dialog_seq(data, config.backward_size)
            if self.seq
            else self.flatten_dialog(data, config.backward_size)
    	)
        self.data_size = len(self.data)
        self.indexes = list(range(self.data_size))
        #

    def flatten_dialog(self, data, backward_size):
        """ Turns a list of dialogs into a flat list of context, response tuples. """
        results = []
        for dlg in data:
            goal = dlg.goal
            for i in range(1, len(dlg.dlg)):
                if dlg.dlg[i].speaker == USR:
                    continue
                e_idx = i
                s_idx = max(0, e_idx - backward_size)
                response = dlg.dlg[i].copy()
                response['utt'] = self.pad_to(self.max_utt_len, response.utt, do_pad=False)
                context = []
                for turn in dlg.dlg[s_idx: e_idx]:
                    turn['utt'] = self.pad_to(self.max_utt_len, turn.utt, do_pad=False)
                    context.append(turn)
                results.append(Pack(context=context, response=response, goal=goal))
        return results

    def flatten_dialog_seq(self, data, backward_size):
        """
        Turn each dialog in list of dialogs into a list of context, response pairs.

        Backward_size indicates how many previous utterances to condition on.
        This should be limited to 1 or 2 at most limiting dependencies.

        The speaker is SYS, so USR utterances are not modeled.
        """
        results = []
        for dlg in data:
            goal = dlg.goal
            context_responses = []
            for i in range(1, len(dlg.dlg)):
                if dlg.dlg[i].speaker == USR:
                    continue
                e_idx = i
                s_idx = max(0, e_idx - backward_size)
                response = dlg.dlg[i].copy()
                response['utt'] = self.pad_to(self.max_utt_len, response.utt, do_pad=False)
                context = []
                for turn in dlg.dlg[s_idx: e_idx]:
                    turn['utt'] = self.pad_to(self.max_utt_len, turn.utt, do_pad=False)
                    context.append(turn)
                context_responses.append(Pack(context=context, response=response, goal=goal))
            results.append(context_responses)
        return results


    def epoch_init(self, config, shuffle=True, verbose=True, fix_batch=False):
        super(DealDataLoaders, self).epoch_init(config, shuffle=shuffle, verbose=verbose)

    def _prepare_batch(self, selected_index):
        return (self._prepare_batch_seq if self.seq else self._prepare_batch_flat)(selected_index)
        #return (self._prepare_batch_seq_dlg if self.seq else self._prepare_batch_flat)(selected_index)

    def _prepare_batch_flat(self, selected_index):
        rows = [self.data[idx] for idx in selected_index]

        ctx_utts, ctx_lens = [], []
        out_utts, out_lens = [], []
        goals, goal_lens = [], []

        for row in rows:
            in_row, out_row, goal_row = row.context, row.response, row.goal

            # source context
            batch_ctx = []
            for turn in in_row:
                batch_ctx.append(self.pad_to(self.max_utt_len, turn.utt, do_pad=True))
            ctx_utts.append(batch_ctx)
            ctx_lens.append(len(batch_ctx))

            # target response
            out_utt = [t for idx, t in enumerate(out_row.utt)]
            out_utts.append(out_utt)
            out_lens.append(len(out_utt))

            # goal
            goals.append(goal_row)
            goal_lens.append(len(goal_row))

        vec_ctx_lens = np.array(ctx_lens) # (batch_size, ), number of turns
        max_ctx_len = np.max(vec_ctx_lens)
        vec_ctx_utts = np.zeros((self.batch_size, max_ctx_len, self.max_utt_len), dtype=np.int32)
        # confs is used to add some hand-crafted features
        vec_ctx_confs = np.ones((self.batch_size, max_ctx_len), dtype=np.float32)
        vec_out_lens = np.array(out_lens) # (batch_size, ), number of tokens
        max_out_len = np.max(vec_out_lens)
        vec_out_utts = np.zeros((self.batch_size, max_out_len), dtype=np.int32)

        max_goal_len, min_goal_len = max(goal_lens), min(goal_lens)
        if max_goal_len != min_goal_len or max_goal_len != 6:
            print('FATAL ERROR!')
            exit(-1)
        self.goal_len = max_goal_len
        vec_goals = np.zeros((self.batch_size, self.goal_len), dtype=np.int32)

        for b_id in range(self.batch_size):
            vec_ctx_utts[b_id, :vec_ctx_lens[b_id], :] = ctx_utts[b_id]
            vec_out_utts[b_id, :vec_out_lens[b_id]] = out_utts[b_id]
            vec_goals[b_id, :] = goals[b_id]

        return Pack(context_lens=vec_ctx_lens, \
                    contexts=vec_ctx_utts, \
                    context_confs=vec_ctx_confs, \
                    output_lens=vec_out_lens, \
                    outputs=vec_out_utts, \
                    goals=vec_goals)

    def _prepare_batch_seq(self, selected_index):
        dlgs = [self.data[idx] for idx in selected_index]

        dlg_idxs, dlg_lens = [], []
        ctx_utts, ctx_lens = [], []
        out_utts, out_lens = [], []
        goals, goal_lens = [], []
        partner_goals_list, num_partner_goals = [], []
        partitions, num_partitions = [], []

        # flatten dialogs here
        # keep pointers
        for i, rows in enumerate(dlgs):
            dlg_len = 0
            for row in rows:
                in_row, out_row, goal_row = row.context, row.response, row.goal

                # source context
                batch_ctx = []
                for turn in in_row:
                    batch_ctx.append(self.pad_to(self.max_utt_len, turn.utt, do_pad=True))
                ctx_utts.append(batch_ctx)
                ctx_lens.append(len(batch_ctx))

                # target response
                out_utt = [t for idx, t in enumerate(out_row.utt)]
                out_utts.append(out_utt)
                out_lens.append(len(out_utt))

                # goal
                goals.append(goal_row)
                goal_lens.append(len(goal_row))

                # valid partner goals
                partner_goals = get_valid_contexts_ints(goal_row)
                partner_goals_list.append(partner_goals)
                num_partner_goals.append(len(partner_goals))

                # partitions
                _partitions = get_latent_powerset(goal_row)
                # list of list of tuples, each tuple is a goal
                # and the inner list represents all possible partner goals
                partitions.append(_partitions)
                num_partitions.append(len(_partitions))

                # dialog index for getting features in sequence model
                dlg_idxs.append(i)

                dlg_len += 1

            dlg_lens.append(dlg_len)

        effective_batch_size = len(goals)

        vec_ctx_lens = np.array(ctx_lens) # (batch_size, ), number of turns
        max_ctx_len = np.max(vec_ctx_lens)
        vec_ctx_utts = np.zeros((effective_batch_size, max_ctx_len, self.max_utt_len), dtype=np.int32)
        # confs is used to add some hand-crafted features
        vec_ctx_confs = np.ones((effective_batch_size, max_ctx_len), dtype=np.float32)
        vec_out_lens = np.array(out_lens) # (batch_size, ), number of tokens
        max_out_len = np.max(vec_out_lens)
        vec_out_utts = np.zeros((effective_batch_size, max_out_len), dtype=np.int32)

        max_goal_len, min_goal_len = max(goal_lens), min(goal_lens)
        if max_goal_len != min_goal_len or max_goal_len != 6:
            print('FATAL ERROR!')
            exit(-1)
        self.goal_len = max_goal_len
        vec_goals = np.zeros((effective_batch_size, self.goal_len), dtype=np.int32)

        max_partner_goals = max(num_partner_goals)
        vec_partner_goals = np.zeros(
            (effective_batch_size, max_partner_goals, self.goal_len),
            dtype=np.int32,
        )
        vec_num_partner_goals = np.array(num_partner_goals)

        # just always pad to 128, makes things easier
        #max_partitions = max(num_partitions)
        max_partitions = 128
        vec_partitions = np.zeros(
            # 3 item types
            (effective_batch_size, max_partitions, 3),
            dtype=np.int32,
        )
        vec_num_partitions = np.array(num_partitions)

        vec_dlg_idxs = np.array(dlg_idxs, dtype=np.int32)
        vec_dlg_lens = np.array(dlg_lens, dtype=np.int32)

        # 
        for b_id in range(effective_batch_size):
            vec_ctx_utts[b_id, :vec_ctx_lens[b_id], :] = ctx_utts[b_id]
            vec_out_utts[b_id, :vec_out_lens[b_id]] = out_utts[b_id]
            vec_goals[b_id, :] = goals[b_id]
            for pg_id in range(num_partner_goals[b_id]):
                vec_partner_goals[b_id, pg_id, :] = partner_goals_list[b_id][pg_id]
            for p_id in range(num_partitions[b_id]):
                vec_partitions[b_id, p_id, :] = partitions[b_id][p_id]

        return Pack(
            context_lens = vec_ctx_lens, 
            contexts = vec_ctx_utts, 
            context_confs = vec_ctx_confs, 
            output_lens = vec_out_lens, 
            outputs = vec_out_utts, 
            goals = vec_goals,
            partner_goals = vec_partner_goals,
            num_partner_goals = vec_num_partner_goals,
            dlg_idxs = vec_dlg_idxs,
            dlg_lens = vec_dlg_lens,
            partitions = vec_partitions,
            num_partitions = vec_num_partitions,
        )

