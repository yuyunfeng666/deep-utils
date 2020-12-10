def log_sum_exp(tensor):
    max_score, _ = torch.max(tensor, -1)
    max_score_broadcast = max_score.view(
        torch.Size(torch.cat([torch.tensor(max_score.size()).long(), torch.tensor([-1], dtype=torch.long)]))
    ).expand(tensor.size())

    score = max_score + \
        torch.log(torch.sum(torch.exp(tensor - max_score_broadcast), -1))

    # print('==============max_score==============')
    # print(max_score.size())
    # print(max_score)
    # print('====max_score_broadcast====')
    # print(max_score_broadcast.size())
    # print(max_score_broadcast)
    # print('===========score========')
    # print(score.size())
    # print(score)

    return score


class Crf(nn.Module):
    def __init__(self, tag2idx):
        super(Crf, self).__init__()

        self.tag2idx = tag2idx
        self.tag_size = len(self.tag2idx)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tag_size, self.tag_size)
        )

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[self.tag2idx[START_TAG], :] = -10000
        self.transitions.data[:, self.tag2idx[STOP_TAG]] = -10000

    def _batch_forward_alg(self, packed_batch_features):
        batch_features, batch_lens = pad_packed_sequence(
            packed_batch_features, batch_first=True
        )
        # features_batch: (tokens_size, batch_size, token_size)
        features_batch = batch_features.transpose(0, 1)
        tokens_size = features_batch.size(0)
        batch_size = features_batch.size(1)

        '''calculate the mask matrix about score from log sum exp'''

        # expanded_batch_lens: (batch_size, tokens_size)
        expanded_batch_lens = batch_lens.view(-1, 1).expand(batch_size, tokens_size)
        # expanded_batch_arange: (batch_size, tokens_size)
        expanded_batch_arange = torch.arange(tokens_size, dtype=torch.long).view(1, -1).expand(
            batch_size, tokens_size
        )
        # mask_batch_matrix: (batch_size, tokens_size)
        mask_batch_matrix = torch.lt(expanded_batch_arange, expanded_batch_lens).float()
        # expanded_mask_batch_matrix: (batch_size, tokens_size, tag_size)
        expanded_mask_batch_matrix = mask_batch_matrix.view(
            batch_size, tokens_size, -1
        ).expand(
            batch_size, tokens_size, self.tag_size
        )
        # expanded_mask_matrix_batch: (tokens_size, batch_size, tag_size)
        expanded_mask_matrix_batch = expanded_mask_batch_matrix.transpose(0, 1).cuda()

        # print('==============expanded_batch_lens==============')
        # print(expanded_batch_lens.size())
        # print(expanded_batch_lens)
        # print('====expanded_batch_arange====')
        # print(expanded_batch_arange.size())
        # print(expanded_batch_arange)
        # print('===========mask_batch_matrix========')
        # print(mask_batch_matrix.size())
        # print(mask_batch_matrix)
        # print('===========expanded_mask_batch_matrix========')
        # print(expanded_mask_batch_matrix.size())
        # print(expanded_mask_batch_matrix)
        # print('===========expanded_mask_matrix_batch========')
        # print(expanded_mask_matrix_batch.size())
        # print(expanded_mask_matrix_batch)

        batch_forward_score = torch.full((batch_size, self.tag_size), -10000.).cuda()
        batch_forward_score[:, self.tag2idx[START_TAG]] = 0.

        for batch_feature, batch_mask_matrix in zip(features_batch, expanded_mask_matrix_batch):
            # batch_feature: (batch_size, tag_size)

            # expanded_batch_forward_score: (batch_size, tag_size, tag_size)
            expanded_batch_forward_score = batch_forward_score.view(
                batch_size, -1, self.tag_size
            ).expand(batch_size, self.tag_size, self.tag_size)
            # expanded_batch_emit_score: (batch_size, tag_size, tag_size)
            expanded_batch_emit_score = batch_feature.view(
                batch_size, self.tag_size, -1
            ).expand(batch_size, self.tag_size, self.tag_size)
            # expanded_batch_transition_score: (batch_size, tag_size, tag_size)
            expanded_batch_transition_score = self.transitions.view(
                1, self.tag_size, self.tag_size
            ).expand(batch_size, self.tag_size, self.tag_size)
            # expanded_batch_next_tag_score: (batch_size, tag_size, tag_size)
            expanded_batch_next_tag_score = expanded_batch_forward_score + expanded_batch_emit_score +\
                expanded_batch_transition_score

            # print('==============batch_forward_score==============')
            # print(batch_forward_score.size())
            # print(batch_forward_score)
            # print('==============expanded_batch_forward_score==============')
            # print(expanded_batch_forward_score.size())
            # print(expanded_batch_forward_score)
            # print('====expanded_batch_emit_score====')
            # print(expanded_batch_emit_score.size())
            # print(expanded_batch_emit_score)
            # print('===========expanded_batch_transition_score========')
            # print(expanded_batch_transition_score.size())
            # print(expanded_batch_transition_score)
            # print('===========expanded_batch_next_tag_score========')
            # print(expanded_batch_next_tag_score.size())
            # print(expanded_batch_next_tag_score)

            # batch_log_sum_exp_score: (batch_size, tag_size)
            batch_log_sum_exp_score = log_sum_exp(expanded_batch_next_tag_score)
            # batch_forward_score: (batch_size, tag_size)
            batch_forward_score = batch_mask_matrix * batch_log_sum_exp_score + \
                (1 - batch_mask_matrix) * batch_forward_score

            # print('===========batch_log_sum_exp_score========')
            # print(batch_log_sum_exp_score.size())
            # print(batch_log_sum_exp_score)
            # print('===========batch_mask_matrix========')
            # print(batch_mask_matrix.size())
            # print(batch_mask_matrix)
            # print('===========batch_forward_score========')
            # print(batch_forward_score.size())
            # print(batch_forward_score)
            # import sys
            # sys.exit(-1)
        # batch_stop_score: (batch_size, tag_size)
        batch_stop_score = self.transitions[self.tag2idx[STOP_TAG]].view(
            1, self.tag_size
        ).expand(batch_size, self.tag_size)
        # batch_terminal_score: (batch_size, tag_size)
        batch_terminal_score = batch_forward_score + batch_stop_score
        # batch_terminal_batch_log_sum_exp_score
        batch_terminal_batch_log_sum_exp_score = log_sum_exp(batch_terminal_score)
        # print('===========batch_stop_score========')
        # print(batch_stop_score.size())
        # print(batch_stop_score)
        # print('===========batch_terminal_score========')
        # print(batch_terminal_score.size())
        # print(batch_terminal_score)
        # print('===========batch_terminal_batch_log_sum_exp_score========')
        # print(batch_terminal_batch_log_sum_exp_score.size())
        # print(batch_terminal_batch_log_sum_exp_score)

        return torch.sum(batch_terminal_batch_log_sum_exp_score)

    def _batch_score_features_tags(self, packed_batch_features, batch_tags):
        batch_features, batch_lens = pad_packed_sequence(
            packed_batch_features, batch_first=True
        )
        # batch_features: (batch_size, tokens_size, tag_size)
        # batch_tags: (batch_size, tokens_size)
        batch_size = batch_features.size(0)
        tokens_size = batch_features.size(1)

        # mask: (batch_size, tokens_size, tag_size)
        mask = torch.lt(
            torch.arange(tokens_size, dtype=torch.long).view(1, -1).expand(batch_size, tokens_size).cuda(),
            batch_lens.view(-1, 1).expand(batch_size, tokens_size).cuda()
        ).float().view(batch_size, tokens_size, -1).expand(batch_size, tokens_size, self.tag_size)

        # batch_transitions: (batch_size, tag_size)
        batch_transitions = self.transitions.view(
            1, self.tag_size, self.tag_size
        ).expand(batch_size, self.tag_size, self.tag_size)

        # batch_arange: (batch_size, tokens_size, tag_size)
        batch_arange = torch.arange(self.tag_size, dtype=torch.long).view(1, -1).expand(
            tokens_size, self.tag_size
        ).view(1, tokens_size, self.tag_size).expand(
            batch_size, tokens_size, self.tag_size
        ).cuda()

        # print('=============batch_features=============')
        # print(batch_features.size())
        # print(batch_features)
        # print('=============batch_transitions=============')
        # print(batch_transitions.size())
        # print(batch_transitions)
        # print('=============batch_arange=============')
        # print(batch_arange.size())
        # print(batch_arange)
        # print('=============mask=============')
        # print(mask.size())
        # print(mask)
        # print('=============batch_tags=============')
        # print(batch_tags.size())
        # print(batch_tags)

        # expanded_batch_tags:(batch_size, tokens_size, tag_size)
        expanded_batch_tags = batch_tags.view(batch_size, tokens_size, -1).expand(
            batch_size, tokens_size, self.tag_size
        )
        # batch_transitions_exchange_matrix: (batch_size, tokens_size, tag_size)
        batch_transitions_exchange_matrix = torch.eq(
            expanded_batch_tags, batch_arange
        ).float()

        '''calculate the score about batch transitions'''

        # exchanged_batch_transitions: (batch_size, tokens_size, tag_size)
        exchanged_batch_transitions = torch.matmul(
            batch_transitions_exchange_matrix, batch_transitions
        )
        # batch_tags_with_start_tag: (batch_size, tokens_size, tag_size)
        batch_tags_with_start_tag = torch.cat([
            torch.tensor(
                [[self.tag2idx[START_TAG]]], dtype=torch.long
            ).expand(batch_size, 1).cuda(),
            batch_tags[:, :-1]
        ], dim=1).view(batch_size, tokens_size, -1).expand(
            batch_size, tokens_size, self.tag_size
        )
        # batch_transitions_select_matrix: (batch_size, tokens_size, tag_size)
        batch_transitions_select_matrix = torch.eq(
            batch_tags_with_start_tag, batch_arange
        ).float()
        # batch_transitions_score_matrix: (batch_size, tokens_size, tag_size)
        batch_transitions_score_matrix = exchanged_batch_transitions * batch_transitions_select_matrix
        # masked_batch_transitions_score_matrix: (batch_size, tokens_size, tag_size)
        masked_batch_transitions_score_matrix = batch_transitions_score_matrix * mask

        batch_transitions_score = torch.sum(masked_batch_transitions_score_matrix)

        # print('============expanded_batch_tags===========')
        # print(expanded_batch_tags.size())
        # print(expanded_batch_tags)
        # print('=============batch_transitions_exchange_matrix=============')
        # print(batch_transitions_exchange_matrix.size())
        # print(batch_transitions_exchange_matrix)
        # print('============exchanged_batch_transitions===========')
        # print(exchanged_batch_transitions.size())
        # print(exchanged_batch_transitions)
        # print('============batch_tags_with_start_tag===========')
        # print(batch_tags_with_start_tag.size())
        # print(batch_tags_with_start_tag)
        # print('============batch_transitions_select_matrix===========')
        # print(batch_transitions_select_matrix.size())
        # print(batch_transitions_select_matrix)
        # print('============batch_transitions_score_matrix===========')
        # print(batch_transitions_score_matrix.size())
        # print(batch_transitions_score_matrix)
        # print('============masked_batch_transitions_score_matrix===========')
        # print(masked_batch_transitions_score_matrix.size())
        # print(masked_batch_transitions_score_matrix)

        '''calculate the score about batch features'''

        # batch_features_score_matrix: (batch_size, tokens_size, tag_size)
        batch_features_score_matrix = batch_features * batch_transitions_exchange_matrix

        batch_features_score = torch.sum(batch_features_score_matrix)

        # print('============batch_features_score_matrix===========')
        # print(batch_features_score_matrix.size())
        # print(batch_features_score_matrix)

        '''calculate the score about stop tags'''

        stop_batch_tag = torch.sum(
            batch_tags * torch.eq(
                batch_lens.view(-1, 1).expand(batch_size, tokens_size),
                torch.arange(1, tokens_size + 1, dtype=torch.long).view(1, -1).expand(batch_size, tokens_size)
            ).long().cuda(),
            dim=1
        )
        stop_batch_transitions_select_matrix = torch.eq(
            stop_batch_tag.view(-1, 1).expand(batch_size, self.tag_size),
            torch.arange(self.tag_size).view(1, -1).expand(
                batch_size, self.tag_size
            ).cuda()
        ).float()
        expand_stop_transitions = self.transitions[
            self.tag2idx[STOP_TAG]
        ].view(1, -1).expand(batch_size, self.tag_size)

        stop_scores = expand_stop_transitions * stop_batch_transitions_select_matrix

        # print('==============stop_batch_tag==============')
        # print(stop_batch_tag.size())
        # print(stop_batch_tag)
        # print('====stop_batch_transitions_select_matrix====')
        # print(stop_batch_transitions_select_matrix.size())
        # print(stop_batch_transitions_select_matrix)
        # print('===========expand_stop_transitions========')
        # print(expand_stop_transitions.size())
        # print(expand_stop_transitions)
        # print('============stop_scores===========')
        # print(stop_scores.size())
        # print(stop_scores)

        batch_score = batch_transitions_score + batch_features_score + torch.sum(stop_scores)
        # import sys
        # sys.exit(-1)
        return batch_score

    def forward(self, features):
        """viterbi_decode"""
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tag_size), -10000.)
        if torch.cuda.is_available():
            init_vvars = init_vvars.cuda()
        init_vvars[0][self.tag2idx[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feature in features:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tag_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = torch.argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feature).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag2idx[STOP_TAG]]
        best_tag_id = torch.argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag2idx[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def loss(self, packed_batch_features, packed_batch_tag_idxes):
        batch_forward_score = self._batch_forward_alg(packed_batch_features)
        padded_batch_tag_idxes, _ = pad_packed_sequence(
            sequence=packed_batch_tag_idxes, batch_first=True
        )
        batch_gold_score = self._batch_score_features_tags(
            packed_batch_features, padded_batch_tag_idxes
        )

        return (batch_forward_score - batch_gold_score) / len(padded_batch_tag_idxes)


class Crf2(nn.Module):
    def __init__(self, tag_size):
        super(Crf2, self).__init__()
        self.tag_size = tag_size
        self.transitions = nn.Parameter(torch.zeros(self.tag_size, self.tag_size))

    @staticmethod
    def _logsumexp(vec):
        max_score, _ = vec.max(1)
        return max_score + torch.log(torch.exp(vec - max_score.unsqueeze(1)).sum(1))

    def _forward_alg(self, feats, feats_mask):
        _zeros = torch.zeros_like(feats[0]).to(feats.device)
        state = torch.where(feats_mask[0].view(-1, 1), feats[0], _zeros)
        transition_params = self.transitions.unsqueeze(0)
        for i in range(1, feats.shape[0]):
            transition_scores = state.unsqueeze(2) + transition_params
            new_state = feats[i] + Crf2._logsumexp(transition_scores)
            state = torch.where(feats_mask[i].view(-1, 1), new_state, state)
        all_mask = feats_mask.any(0).float()
        return Crf2._logsumexp(state) * all_mask

    def _score_sentence(self, feats, tags, feats_mask):
        # Gives the score of a provided tag sequence
        feats_mask = feats_mask.float()
        time_step, batch_size, tags_size = feats.shape
        s_score = feats.view(-1, tags_size).gather(1, tags.view(-1, 1)) * feats_mask.view(-1, 1)
        u_score = s_score.view(-1, batch_size).sum(0)
        if time_step > 1:
            t_mask = feats_mask[:-1].view(-1, 1) * feats_mask[1:].view(-1, 1)
            t_scores = self.transitions.index_select(0, tags[0:-1].view(-1))
            t_score = t_scores.gather(1, tags[1:].view(-1, 1)) * t_mask
            u_score += t_score.view(-1, batch_size).sum(0)
        return u_score

    def loss(self, feats, tags, feats_len):
        # feats is [batch,time,tag_size] float tensor
        # tags is [batch,time]  tensor
        # feats_len is [batch,]  tensor
        feats = feats.transpose(0, 1).contiguous()
        tags = tags.long().transpose(0, 1).contiguous()
        base_index = torch.arange(0, feats.shape[0]).unsqueeze(0).expand(feats.shape[1], -1).to(feats.device)
        feats_mask = base_index < feats_len.long().view(-1, 1)
        feats_mask = feats_mask.transpose(0, 1).contiguous()

        forward_score = self._forward_alg(feats, feats_mask)
        gold_score = self._score_sentence(feats, tags, feats_mask)

        return forward_score - gold_score
