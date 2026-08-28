from experiments.figure10.streaming_conformance import prompt_token_count


class BatchEncodingLike:
    def __init__(self, input_ids):
        self.data = {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}

    def keys(self):
        return self.data.keys()

    def __contains__(self, key):
        return key in self.data

    def __getitem__(self, key):
        return self.data[key]


def test_prompt_token_count_reads_batch_encoding_input_ids():
    assert prompt_token_count(BatchEncodingLike(list(range(30)))) == 30


def test_prompt_token_count_accepts_plain_token_list():
    assert prompt_token_count(list(range(30))) == 30


def test_prompt_token_count_unwraps_single_batch():
    assert prompt_token_count({"input_ids": [list(range(30))]}) == 30
