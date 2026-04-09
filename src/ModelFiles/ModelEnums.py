from enum import Enum

class LSTMModelType(str, Enum):
    LSTM = "lstm"
    BILSTM = "bilstm"
    MULTIHEAD_ATTENTION_LSTM = "multihead_attention_lstm"
    MULTIHEAD_ATTENTION_BILSTM = "multihead_attention_bilstm"

class TransformerModelType(str, Enum):
    PATCHTST = "patchtst"
    ITRANSFORMER = "itransformer"
    TIMEXER = "timexer"