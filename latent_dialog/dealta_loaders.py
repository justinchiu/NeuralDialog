import numpy as np
from latent_dialog.utils import Pack
from latent_dialog.base_data_loaders import BaseDataLoaders
from latent_dialog.corpora import USR, SYS
import json


class DealDataLoaders(BaseDataLoaders):
    def __init__(self, name, data, config):
        super(DealDataLoaders, self).__init__(name)
        self.max_utt_len = config.max_utt_len
        self.data = self.flatten_dialog(data, config.backward_size)
        # hopefully backward size being infinite is ok
        self.data_long = self.flatten_dialong(data, config.backward_size)
        self.data_size = len(self.data)
        self.indexes = list(range(self.data_size))

    def flatten_dialog(self, data, backward_size):
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

    def flatten_dialong(self, data, backward_size):
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
                import pdb; pdb.set_trace()
        return results


    def epoch_init(self, config, shuffle=True, verbose=True, fix_batch=False):
        super(DealDataLoaders, self).epoch_init(config, shuffle=shuffle, verbose=verbose)

    def _prepare_batch(self, selected_index):
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
            import pdb; pdb.set_trace()

        vec_ctx_lens = np.array(ctx_lens) # (batch_size, ), number of turns
        max_ctx_len = np.max(vec_ctx_lens)
        vec_ctx_utts = np.zeros((self.batch_size, max_ctx_len, self.max_utt_len), dtype=np.int32)
        # confs is used to add some hand-crafted features
        vec_ctx_confs = np.ones((self.batch_size, max_ctx_len), dtype=np.float32)
        vec_out_lens = np.array(out_lens) # (batch_size, ), number of tokens
        max_out_len = np.max(vec_out_lens)
        vec_out_utts = np.zeros((self.batch_size, max_out_len), dtype=np.int32)
        import pdb; pdb.set_trace()

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

